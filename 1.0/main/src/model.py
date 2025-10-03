########################################################################################################
# RunningWay Large Language Model
# Github: https://github.com/QWXL/RunningWay
########################################################################################################

import os, math, gc, importlib
import sys
import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
from pytorch_lightning.strategies import DeepSpeedStrategy
import deepspeed 
if importlib.util.find_spec('deepspeed'):
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from typing import Optional
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rnw.runningway_config import RunningWayConfig
from rnw.state_pool import StateMemoryPool

try:
    print('RNW_MY_TESTING', os.environ["RNW_MY_TESTING"])
except:
    os.environ["RNW_MY_TESTING"] = 'r010'

# 设置 RNW_JIT_ON 环境变量的默认值
os.environ.setdefault("RNW_JIT_ON", "0")

def __nop(ob):
    return ob


MyModule = nn.Module
MyFunction = __nop
if os.environ["RNW_JIT_ON"] == "1":
    MyModule = torch.jit.ScriptModule
    MyFunction = torch.jit.script_method


########################################################################################################
# CUDA Kernel
########################################################################################################

from torch.utils.cpp_extension import load
os.environ.setdefault("RNW_HEAD_SIZE", "64")
os.environ.setdefault("RNW_MY_TESTING", "r010")

HEAD_SIZE = int(os.environ["RNW_HEAD_SIZE"])

if 'r010' in os.environ["RNW_MY_TESTING"]:
    print(f"[RNW] Load Cuda Kernel with {os.environ["RNW_MY_TESTING"]}")
    CHUNK_LEN = 16

    # 支持 fused_state，需要等待Cuda内核就绪
    flags = ['-res-usage', f'-D_C_={HEAD_SIZE}', f"-D_CHUNK_LEN_={CHUNK_LEN}", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"]
    load(name="wind_backstepping", sources=[f'cuda/wkv7_cuda.cu', 'cuda/wkv7_op.cpp'], is_python_module=False, verbose=True, extra_cuda_cflags=flags)
    print(f"[RNW] Successfully loaded wind_backstepping kernel with HEAD_SIZE={HEAD_SIZE}, CHUNK_LEN={CHUNK_LEN}")


    class WindBackstepping_original(torch.autograd.Function):
        @staticmethod
        def forward(ctx, w,q,k,v,z,b):
            B,T,H,C = w.shape 
            print(f"[WindBackstepping_original] Forward: B={B}, T={T}, H={H}, C={C}")
            assert T%CHUNK_LEN == 0 # if T%CHUNK_LEN != 0: pad your input to T%CHUNK_LEN == 0, or change CHUNK_LEN (will be slower)
            # assert all(i.dtype==torch.bfloat16 for i in [w,q,k,v,z,b])
            assert all(i.is_contiguous() for i in [w,q,k,v,z,b])
            y = torch.empty_like(v)
            s = torch.empty(B,H,T//CHUNK_LEN,C,C, dtype=torch.float32,device=w.device)
            sa = torch.empty(B,T,H,C, dtype=torch.float32,device=w.device)
            torch.ops.wind_backstepping.forward(w,q,k,v,z,b, y,s,sa)
            ctx.save_for_backward(w,q,k,v,z,b,s,sa)
            print(f"[WindBackstepping_original] Forward completed")
            return y
        @staticmethod
        def backward(ctx, dy):
            print(f"[WindBackstepping_original] Backward started")
            # assert all(i.dtype==torch.bfloat16 for i in [dy])
            assert all(i.is_contiguous() for i in [dy])
            w,q,k,v,z,b,s,sa = ctx.saved_tensors
            dw,dq,dk,dv,dz,db = [torch.empty_like(x) for x in [w,q,k,v,z,b]]
            torch.ops.wind_backstepping.backward(w,q,k,v,z,b, dy,s,sa, dw,dq,dk,dv,dz,db)
            print(f"[WindBackstepping_original] Backward completed")
            return dw,dq,dk,dv,dz,db


    class WindBackstepping(torch.autograd.Function):
        @staticmethod
        def forward(ctx, w, q, k, v, z, b, fused_state=None):
            B, T, H, C = w.shape 
            print(f"[WindBackstepping] Forward: B={B}, T={T}, H={H}, C={C}")
            assert T % CHUNK_LEN == 0
        
            # 处理 fused_state
            if fused_state is not None:
                assert fused_state.shape == (B, H, C)
                assert fused_state.is_contiguous()
                print(f"[WindBackstepping] Using provided fused_state with shape {fused_state.shape}")
            else:
                # 如果没有提供 fused_state，创建一个零状态
                fused_state = torch.zeros(B, H, C, dtype=torch.float32, device=w.device)
                print(f"[WindBackstepping] Created zero fused_state with shape {fused_state.shape}")
        
            y = torch.empty_like(v)
            s = torch.empty(B, H, T // CHUNK_LEN, C, C, dtype=torch.float32, device=w.device)
            sa = torch.empty(B, T, H, C, dtype=torch.float32, device=w.device)
        
            # 调用修改后的 CUDA 内核，传入 fused_state
            torch.ops.wind_backstepping.forward_with_state(w, q, k, v, z, b, fused_state, y, s, sa)
        
            ctx.save_for_backward(w, q, k, v, z, b, fused_state, s, sa)
            print(f"[WindBackstepping] Forward completed")
            return y

        @staticmethod
        def backward(ctx, dy):
            print(f"[WindBackstepping] Backward started")
            w, q, k, v, z, b, fused_state, s, sa = ctx.saved_tensors
            dw, dq, dk, dv, dz, db = [torch.empty_like(x) for x in [w, q, k, v, z, b]]
            dfused_state = torch.empty_like(fused_state) if fused_state is not None else None

            torch.ops.wind_backstepping.backward_with_state(
                w, q, k, v, z, b, fused_state, dy, s, sa, 
                dw, dq, dk, dv, dz, db, dfused_state
            )
            print(f"[WindBackstepping] Backward completed")
            return dw, dq, dk, dv, dz, db, dfused_state

    # 保持向后兼容的原始函数
    def RUN_CUDA_RWKV7g_original(q, w, k, v, a, b):
        B,T,HC = q.shape
        print(f"[RUN_CUDA_RWKV7g_original] Input shape: q={q.shape}, w={w.shape}, k={k.shape}, v={v.shape}, a={a.shape}, b={b.shape}")
        q,w,k,v,a,b = [i.view(B,T,HC//64,64) for i in [q,w,k,v,a,b]]
        result = WindBackstepping_original.apply(w,q,k,v,a,b).view(B,T,HC)
        print(f"[RUN_CUDA_RWKV7g_original] Completed")
        return result

    USE_MULTI_STATE = True  # 设置为 False 可回退到原始行为
    USE_NEW_CUDA_KERNEL = False  # 初始为 False，等待新内核就绪
    print(f"[RNW] USE_MULTI_STATE={USE_MULTI_STATE}, USE_NEW_CUDA_KERNEL={USE_NEW_CUDA_KERNEL}")

    # 兼容性包装器
    def RUN_CUDA_RWKV7g_compatible(q, w, k, v, a, b, fused_state=None, state_gate_weights=None):
        """
        兼容性包装器，支持多状态融合的渐进式部署
        """
        B, T, HC = q.shape
        H = HC // 64
        print(f"[RUN_CUDA_RWKV7g_compatible] Input shapes: q={q.shape}, w={w.shape}, k={k.shape}, v={v.shape}, a={a.shape}, b={b.shape}")
        if fused_state is not None:
            print(f"[RUN_CUDA_RWKV7g_compatible] fused_state shape: {fused_state.shape}")
        
        # 如果不使用多状态，直接调用原始函数
        if not USE_MULTI_STATE or fused_state is None:
            print(f"[RUN_CUDA_RWKV7g_compatible] Using original function (USE_MULTI_STATE={USE_MULTI_STATE}, fused_state is None={fused_state is None})")
            return RUN_CUDA_RWKV7g_original(q, w, k, v, a, b)
        
        # 如果新 CUDA 内核可用，使用新内核
        if USE_NEW_CUDA_KERNEL and hasattr(torch.ops.wind_backstepping, 'forward_with_state'):
            print(f"[RUN_CUDA_RWKV7g_compatible] Using new CUDA kernel")
            q, w, k, v, a, b = [i.view(B, T, H, 64) for i in [q, w, k, v, a, b]]
            fused_state_reshaped = fused_state.view(B, H, 64)
            result = WindBackstepping.apply(w, q, k, v, a, b, fused_state_reshaped).view(B, T, HC)
            print(f"[RUN_CUDA_RWKV7g_compatible] New CUDA kernel completed")
            return result
        
        # 否则，使用 Python 模拟的多状态融合（训练时可用，推理时较慢）
        print(f"[RUN_CUDA_RWKV7g_compatible] Using fallback implementation")
        return _run_multi_state_fallback(q, w, k, v, a, b, fused_state, state_gate_weights)

    def _run_multi_state_fallback(q, w, k, v, a, b, fused_state, state_gate_weights):
        """
        Python 回退实现：通过多次调用原始 WKV 来模拟多状态融合
        注意：这会增加计算量，但可以验证逻辑正确性
        """
        B, T, HC = q.shape
        H = HC // 64
        print(f"[_run_multi_state_fallback] Input shapes: q={q.shape}, w={w.shape}, k={k.shape}, v={v.shape}, a={a.shape}, b={b.shape}")
        
        # 将 fused_state 分解为三个状态分量
        if state_gate_weights is not None:
            gate_rnn, gate_win, gate_sys = state_gate_weights
            print(f"[_run_multi_state_fallback] Using provided state_gate_weights: rnn={gate_rnn}, win={gate_win}, sys={gate_sys}")
        else:
            gate_rnn, gate_win, gate_sys = 0.33, 0.34, 0.33
            print(f"[_run_multi_state_fallback] Using default state_gate_weights: rnn={gate_rnn}, win={gate_win}, sys={gate_sys}")
        
        # 使用原始 WKV 计算主输出
        main_output = RUN_CUDA_RWKV7g_original(q, w, k, v, a, b)
        print(f"[_run_multi_state_fallback] Original WKV computation completed")
        
        # 模拟状态融合的影响（简化版本）
        # TODO: 应该更精细的操作状态融合
        if fused_state is not None:
            fused_effect = torch.zeros_like(main_output)
            
            # RNN state 影响（长期依赖）
            rnn_effect = fused_state.view(B, 1, HC) * gate_rnn
            fused_effect += rnn_effect.expand(-1, T, -1) * 0.1
            
            # Window state 影响（近期依赖）  
            win_effect = fused_state.view(B, 1, HC) * gate_win
            fused_effect += win_effect.expand(-1, T, -1) * 0.15
            
            # System state 影响（系统提示）
            if gate_sys > 0:
                sys_effect = fused_state.view(B, 1, HC) * gate_sys
                fused_effect += sys_effect.expand(-1, T, -1) * 0.05
            
            main_output = main_output + fused_effect
            print(f"[_run_multi_state_fallback] Applied fused state effect")
        
        print(f"[_run_multi_state_fallback] Completed")
        return main_output




########################################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from rnw.runningway_config import RunningWayConfig

class RNW_Tmix(MyModule):
    def __init__(self, config: RunningWayConfig, layer_id: int):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.my_testing = config.my_testing

        self.head_size = getattr(config, 'head_size', 64)  # 默认值为64
        self.n_head = config.dim_att // self.head_size
        assert config.dim_att % self.n_head == 0
        H = self.n_head
        N = self.head_size
        C = config.n_embd

        # ==================== RunningWay State Gate ====================
        # 门控网络：决定三个state的融合权重 [gate_rnn, gate_win, gate_sys]
        self.state_gate_net = nn.Sequential(
            nn.Linear(C, max(32, C // 4)),  # 轻量MLP
            nn.SiLU(),
            nn.Linear(max(32, C // 4), 3),  # 输出3个权重
            nn.Softmax(dim=-1)
        )
        
        # 状态投影层，将融合状态转换为偏置
        self.state_proj = nn.Linear(C, C, bias=False)
        
        # 初始化默认权重 - 从配置获取
        default_ratios = config.default_state_ratios
        with torch.no_grad():
            self.state_gate_net[2].weight.data.zero_()
            self.state_gate_net[2].bias.data.copy_(
                torch.tensor([default_ratios['rnn'], default_ratios['window'], default_ratios['system']])
            )
            # 状态投影层使用小权重初始化
            nn.init.orthogonal_(self.state_proj.weight, gain=0.1)
        
        # Window state 相关参数 - 从配置获取
        self.window_size = config.window_size
        # ==================== End RunningWay Additions ====================

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (config.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / config.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, C)
            for i in range(C):
                ddd[0, 0, i] = i / C

            self.x_r = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.x_w = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
            self.x_v = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
            self.x_a = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_g = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))

            def ortho_init(x, scale):
                with torch.no_grad():
                    shape = x.shape
                    if len(shape) == 2:
                        gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                        nn.init.orthogonal_(x, gain=gain * scale)
                    elif len(shape) == 3:
                        gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                        for i in range(shape[0]):
                            nn.init.orthogonal_(x[i], gain=gain * scale)
                    else:
                        assert False
                    return x

            www = torch.zeros(C)
            zigzag = torch.zeros(C)
            linear = torch.zeros(C)
            for n in range(C):
                linear[n] = n / (C-1) - 0.5
                zigzag[n] = ((n % N) - ((N-1) / 2)) / ((N-1) / 2)
                zigzag[n] = zigzag[n] * abs(zigzag[n])
                www[n] = -6 + 6 * (n / (C - 1)) ** (1 + 1 * ratio_0_to_1 ** 0.3)

            D_DECAY_LORA = max(32, int(round(  (2.5*(C**0.5))  /32)*32)) # suggestion
            self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
            self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, C), 0.1))
            self.w0 = nn.Parameter(www.reshape(1,1,C) + 0.5 + zigzag*2.5) # !!! 0.5 comes from F.softplus !!!

            D_AAA_LORA = max(32, int(round(  (2.5*(C**0.5))  /32)*32)) # suggestion
            self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
            self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
            self.a0 = nn.Parameter(torch.zeros(1,1,C)-0.19 + zigzag*0.3 + linear*0.4)

            D_MV_LORA = max(32, int(round(  (1.7*(C**0.5))  /32)*32)) # suggestion
            self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
            self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
            self.v0 = nn.Parameter(torch.zeros(1,1,C)+0.73 - linear*0.4)

            # Note: for some data, you can reduce D_GATE_LORA or even remove this gate
            D_GATE_LORA = max(32, int(round(  (5*(C**0.5))  /32)*32)) # suggestion
            self.g1 = nn.Parameter(torch.zeros(C, D_GATE_LORA))
            self.g2 = nn.Parameter(ortho_init(torch.zeros(D_GATE_LORA, C), 0.1))

            self.k_k = nn.Parameter(torch.zeros(1,1,C)+0.71 - linear*0.1)
            self.k_a = nn.Parameter(torch.zeros(1,1,C)+1.02)
            self.r_k = nn.Parameter(torch.zeros(H,N)-0.04)

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            self.receptance = nn.Linear(C, C, bias=False)
            self.key = nn.Linear(C, C, bias=False)
            self.value = nn.Linear(C, C, bias=False)
            self.output = nn.Linear(C, C, bias=False)
            self.ln_x = nn.GroupNorm(H, C, eps=64e-5) # !!! notice eps value !!!

            self.receptance.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            self.key.weight.data.uniform_(-0.05/(C**0.5), 0.05/(C**0.5))
            self.value.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            self.output.weight.data.zero_()

    @MyFunction
    def forward(self, x, v_first, state_pool=None, layer_id=0):
        B, T, C = x.size()
        H = self.n_head
        
        # ==================== RunningWay State Fusion ====================
        use_state_fusion = state_pool is not None and hasattr(state_pool, 'get_state_slices')
        
        if use_state_fusion:
            # 获取三个state
            rnn_state, win_state, sys_state = state_pool.get_state_slices(layer_id)
            
            # 计算门控权重
            x_mean = x.mean(dim=1)  # [B, C]
            gate_weights = self.state_gate_net(x_mean)  # [B, 3]
            gate_rnn, gate_win, gate_sys = gate_weights[:, 0], gate_weights[:, 1], gate_weights[:, 2]
            
            # 扩展门控权重用于广播
            gate_rnn = gate_rnn.view(B, 1, 1)  # [B, 1, 1]
            gate_win = gate_win.view(B, 1, 1)
            gate_sys = gate_sys.view(B, 1, 1)
            
            # 融合三个state [B, H, head_size]
            fused_state = gate_rnn * rnn_state.unsqueeze(0)  # 广播到batch维度
            fused_state = fused_state + gate_win * win_state.unsqueeze(0)
            
            if sys_state is not None:
                fused_state = fused_state + gate_sys * sys_state.unsqueeze(0)
            
            # 将融合状态展平并投影为偏置项 [B, 1, C]
            fused_state_flat = fused_state.reshape(B, C)  # [B, C]
            state_bias = self.state_proj(fused_state_flat).unsqueeze(1)  # [B, 1, C]
        else:
            state_bias = 0.0
            fused_state = None
        # ==================== End State Fusion ====================

        xx = self.time_shift(x) - x

        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.receptance(xr)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5
        k = self.key(xk)
        v = self.value(xv)
        
        # 处理 v_first
        if self.layer_id == 0:
            v_first = v
        else:
            # 简化处理 v_first
            if hasattr(self, 'v_first') and self.v_first is not None:
                v_first_current = self.v_first
            else:
                v_first_current = v
            v = v + (v_first_current - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)
        
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = F.normalize(kk.view(B, T, H, -1), dim=-1, p=2.0).view(B, T, C)
        k = k * (1 + (a - 1) * self.k_a)

        # ==================== WKV 计算与状态偏置 ====================
        # 使用兼容性包装器进行 WKV 计算
        x = RUN_CUDA_RWKV7g_compatible(r, w, k, v, -kk, kk * a, fused_state)
        
        # 应用状态偏置
        x = x + state_bias
        # ==================== End WKV Computation ====================

        x = self.ln_x(x.view(B * T, C)).view(B, T, C)

        x = x + ((r.view(B, T, H, -1) * k.view(B, T, H, -1) * self.r_k).sum(dim=-1, keepdim=True) * v.view(B, T, H, -1)).view(B, T, C)
        x = self.output(x * g)
        
        # ==================== RunningWay State Update ====================
        if use_state_fusion:
            with torch.no_grad():
                # 从当前计算中计算状态更新
                # 使用最后一个token的信息进行状态更新
                last_k = k[:, -1, :].view(B, H, -1)  # [B, H, head_size]
                last_v = v[:, -1, :].view(B, H, -1)  # [B, H, head_size]
                last_r = r[:, -1, :].view(B, H, -1)  # [B, H, head_size]
                
                # 计算状态更新
                # TODO: 添加状态更新逻辑、改用矩阵状态、添加时间衰减等
                state_update = last_k * last_v * last_r
                
                # 使用衰减更新 RNN 状态
                state_pool.update_rnn_state(layer_id, state_update.mean(dim=0), decay=0.9)
                
                # 更新窗口状态
                state_pool.update_window_state(layer_id, state_update.mean(dim=0))
        # ==================== End State Update ====================
        
        # 存储 v_first 供下一层使用
        if self.layer_id == 0:
            self.v_first = v_first
            
        return x, v_first

    def _original_forward(self, x, v_first):
        """回退到原始 RWKV-7 forward，当没有提供 state_pool 时"""
        B, T, C = x.size()
        H = self.n_head
        xx = self.time_shift(x) - x

        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.receptance(xr)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = F.normalize(kk.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)
        k = k * (1 + (a-1) * self.k_a)

        x = RUN_CUDA_RWKV7g_compatible(r, w, k, v, -kk, kk*a)
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)

        x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.r_k).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)
        x = self.output(x * g)
        return x, v_first

    
########################################################################################################

class RWKV_CMix_x070(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0**4))

        self.key = nn.Linear(args.n_embd, args.n_embd * 4, bias=False)
        self.value = nn.Linear(args.n_embd * 4, args.n_embd, bias=False)

        self.key.weight.data.uniform_(-0.5/(args.n_embd**0.5), 0.5/(args.n_embd**0.5))
        self.value.weight.data.zero_()

    @MyFunction
    def forward(self, x):
        xx = self.time_shift(x) - x
        
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2

        return self.value(k)


########################################################################################################
# RunningWay Block (From RWKV Reconstruction)
########################################################################################################

import torch
import torch.nn as nn
from rnw.runningway_config import RunningWayConfig

class Block(nn.Module):
    def __init__(self, config: RunningWayConfig, layer_id: int):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(config.n_embd)

        self.att = RNW_Tmix(config, layer_id)
        self.ffn = RWKV_CMix_x070(config, layer_id)
        
    def forward(self, x, v_first, state_pool=None, layer_id=0):
        """
        Modified forward to support StateMemoryPool
        """
        if self.layer_id == 0:
            x = self.ln0(x)

        # Pass state_pool and layer_id to attention
        x_attn, v_first = self.att(self.ln1(x), v_first, state_pool, layer_id)
        x = x + x_attn

        x = x + self.ffn(self.ln2(x))
        return x, v_first



class L2Wrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss, y):
        ctx.save_for_backward(y)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.saved_tensors[0]
        # to encourage the logits to be close to 0
        factor = 1e-4 / (y.shape[0] * y.shape[1])
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        gy.scatter_(-1, ids, maxx * factor)
        return (grad_output, gy)



class RunningWay(pl.LightningModule):
    def __init__(self, config: RunningWayConfig):
        super().__init__()
        self.config = config
        self.save_hyperparameters(ignore=['config'])  # 保存配置到 checkpoint
        
        # 自动设置默认值（保持与原始代码兼容）
        if not hasattr(config, 'dim_att'):
            config.dim_att = config.n_embd
        if not hasattr(config, 'dim_ffn'):
            config.dim_ffn = int((config.n_embd * 3.5) // 32 * 32)
            
        assert config.n_embd % 32 == 0
        assert config.dim_att % 32 == 0
        assert config.dim_ffn % 32 == 0

        self.emb = nn.Embedding(config.vocab_size, config.n_embd)

        self.blocks = nn.ModuleList([Block(config, i) for i in range(config.n_layer)])

        self.ln_out = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # ==================== RunningWay Integration ====================
        # Calculate model dimensions for StateMemoryPool
        self.n_head = config.dim_att // getattr(config, 'head_size', 64)
        self.head_size = getattr(config, 'head_size', 64)
        
        # State Memory Pool
        self.state_pool = StateMemoryPool(
            total_dim=config.n_embd,
            n_layer=config.n_layer,
            n_head=self.n_head,
            head_size=self.head_size,
            device=None,  # Will be set automatically
            dtype=torch.float32
        )
        
        # State management flags
        self.using_state_pool = config.use_multi_state
        self.current_system_tokens = None
        
        # 应用配置到状态池
        self._apply_config_to_state_pool()
        
        print(f"[RunningWay] Initialized with StateMemoryPool: {config.n_layer} layers, {self.n_head} heads")
        print(f"[RunningWay] Multi-State: {'Enabled' if config.use_multi_state else 'Disabled'}")
        # ==================== End RunningWay Integration ====================

    def _apply_config_to_state_pool(self):
        """应用配置到状态池"""
        config = self.config
        
        # 设置窗口大小
        if hasattr(self.state_pool, 'window_size'):
            self.state_pool.window_size = config.window_size
        
        # 设置默认状态分配比例
        if hasattr(self.state_pool, 'default_state_ratios'):
            self.state_pool.default_state_ratios = config.default_state_ratios.copy()
        
        # 设置当前分配比例
        if hasattr(self.state_pool, 'alpha_sys'):
            self.state_pool.alpha_sys = config.default_state_ratios['system']
            self.state_pool.alpha_rnn = config.default_state_ratios['rnn']
            self.state_pool.alpha_win = config.default_state_ratios['window']

    def configure_optimizers(self):
        config = self.config
        
        lr_decay = set()
        lr_1x = set()
        lr_2x = set()
        for n, p in self.named_parameters():
            if ("att.w0" in n):
                lr_2x.add(n)
            elif (len(p.squeeze().shape) >= 2) and (config.weight_decay > 0) and (".weight" in n):
                lr_decay.add(n)
            else:
                lr_1x.add(n)

        # ==================== RunningWay Optimizer Groups ====================
        # Add StateMemoryPool parameters to appropriate groups
        for n, p in self.named_parameters():
            if "state_pool" in n:
                if "allocator" in n or "state_gate_net" in n:
                    # State gating networks - use standard learning rate
                    lr_1x.add(n)
                elif "system_proj" in n:
                    # System projection - use standard learning rate  
                    lr_1x.add(n)
                else:
                    # State buffers - no weight decay
                    lr_1x.add(n)
        # ==================== End RunningWay Optimizer Groups ====================

        lr_decay = sorted(list(lr_decay))
        lr_1x = sorted(list(lr_1x))
        lr_2x = sorted(list(lr_2x))

        if self.trainer.is_global_zero:
            print('decay', lr_decay, '\n')
            print('1x', lr_1x, '\n')
            print('2x', lr_2x, '\n')

        param_dict = {n: p for n, p in self.named_parameters()}
        
        optim_groups = [
            {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "lr_scale": 1.0},
            {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "lr_scale": 2.0},
        ]

        if self.deepspeed_offload:
            from deepspeed.ops.adam import DeepSpeedCPUAdam
            optimizer = DeepSpeedCPUAdam(
                optim_groups,
                lr=config.lr_init,
                betas=config.betas,
                eps=config.adam_eps,
                bias_correction=True,
                adam_w_mode=(config.weight_decay > 0),
                amsgrad=False
            )
        else:
            from deepspeed.ops.adam import FusedAdam
            optimizer = FusedAdam(
                optim_groups,
                lr=config.lr_init,
                betas=config.betas,
                eps=config.adam_eps,
                bias_correction=True,
                adam_w_mode=(config.weight_decay > 0),
                amsgrad=False
            )

        return optimizer

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if hasattr(strategy, 'config') and 'zero_optimization' in strategy.config:
            cfg = strategy.config["zero_optimization"]
            return cfg.get("offload_optimizer") or cfg.get("offload_param")
        return False

    # ==================== RunningWay State Management ====================
    def set_system_prompt(self, system_tokens: torch.Tensor):
        """Set system prompt for the model"""
        if system_tokens is None:
            self.state_pool.reset_for_no_system()
            self.current_system_tokens = None
        else:
            self.current_system_tokens = system_tokens
            # Encode system tokens to get embeddings
            with torch.no_grad():
                system_emb = self.emb(system_tokens).unsqueeze(0)  # [1, seq_len, n_embd]
                self.state_pool.set_system_prompt(system_emb)
    
    def reset_state(self, keep_system: bool = True):
        """Reset model state while optionally keeping system prompt"""
        self.state_pool.reset_all_states()
        if keep_system and self.current_system_tokens is not None:
            self.set_system_prompt(self.current_system_tokens)
        else:
            self.state_pool.reset_for_no_system()
    
    def enable_state_pool(self):
        """Enable state pool usage"""
        self.using_state_pool = True
        self.config.use_multi_state = True
    
    def disable_state_pool(self):
        """Disable state pool usage (fallback to original RWKV)"""
        self.using_state_pool = False
        self.config.use_multi_state = False
    
    def get_config(self) -> RunningWayConfig:
        """获取当前配置"""
        return self.config
    
    def update_config(self, **kwargs):
        """动态更新配置"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                old_value = getattr(self.config, key)
                setattr(self.config, key, value)
                print(f"[Config] Updated {key}: {old_value} -> {value}")
        
        # 重新应用配置到状态池
        self._apply_config_to_state_pool()
    # ==================== End RunningWay State Management ====================

    def forward(self, idx):
        config = self.config
        B, T = idx.size()
        assert T <= config.ctx_len, "Cannot forward, model ctx_len is exhausted."

        x = self.emb(idx)
        v_first = torch.empty_like(x)

        # ==================== RunningWay Forward Pass ====================
        if self.using_state_pool and self.state_pool is not None:
            # Use state pool for all blocks
            for i, block in enumerate(self.blocks):
                if config.grad_cp == 1:
                    # wrapper 避免非 tensor 参数报错
                    def block_forward(x, v_first):
                        return block(x, v_first, self.state_pool, i)

                    x, v_first = deepspeed.checkpointing.checkpoint(block_forward, x, v_first)
                else:
                    x, v_first = block(x, v_first, self.state_pool, i)
        else:
            # Original RWKV forward without state pool
            for block in self.blocks:
                if config.grad_cp == 1:
                    def block_forward(x, v_first):
                        return block(x, v_first)

                    x, v_first = deepspeed.checkpointing.checkpoint(block_forward, x, v_first)
                else:
                    x, v_first = block(x, v_first)
        # ==================== End RunningWay Forward Pass ====================

        x = self.ln_out(x)
        x = self.head(x)
        return x


    def training_step(self, batch, batch_idx):
        idx, targets = batch
        
        # ==================== RunningWay Training Setup ====================
        # For training, we typically don't want persistent state across batches
        # Reset state at the beginning of each training step
        if self.using_state_pool and self.config.reset_state_per_batch:
            self.reset_state(keep_system=False)
        # ==================== End RunningWay Training Setup ====================
        
        logits = self(idx)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return L2Wrap.apply(loss, logits)

    def training_step_end(self, batch_parts):
        all = self.all_gather(batch_parts)
        if self.trainer.is_global_zero:
            self.trainer.my_loss_all = all

    def generate_init_weight(self):
        print(
            f"""
############################################################################
#
# Init model weight (slow for large models)...
#
############################################################################
"""
        )
        m = {}
        n_params = 0
        
        # ==================== RunningWay Weight Initialization ====================
        # First, initialize StateMemoryPool weights
        state_pool_params = {}
        for n, p in self.state_pool.named_parameters():
            if "allocator" in n:
                # Allocator network - use standard initialization
                if n.endswith('.weight'):
                    nn.init.xavier_uniform_(p)
                else:
                    nn.init.zeros_(p)
            elif "system_proj" in n and n.endswith('.weight'):
                # System projection - use orthogonal initialization
                nn.init.orthogonal_(p, gain=0.1)
            elif "state_gate_net" in n:
                # State gate network - already initialized in RNW_Tmix
                pass
            state_pool_params[n] = p
        
        # Add state pool params to the main state dict
        for n, p in state_pool_params.items():
            m[f"state_pool.{n}"] = p
            print(f"state_pool.{n}: {p.shape} - custom init")
            
        # Add state pool buffers (like rnn_state, window_state, system_state_buffer)
        for n, b in self.state_pool.named_buffers():
            if isinstance(b, torch.Tensor):  # Only include tensor buffers
                m[f"state_pool.{n}"] = b
                print(f"state_pool.{n}: {b.shape} - buffer")
        # ==================== End RunningWay Weight Initialization ====================

        for n in self.state_dict():
            # Skip state_pool parameters and buffers as we've already handled them
            if "state_pool" in n:
                continue
                
            p = self.state_dict()[n]
            shape = p.shape

            s0 = str(shape[0]) if len(shape) > 0 else ""
            s1 = str(shape[1]) if len(shape) > 1 else ""
            s2 = str(shape[2]) if len(shape) > 2 else ""
            s3 = str(shape[3]) if len(shape) > 3 else ""
            print(f"{s0.ljust(5)} {s1.ljust(5)} {s2.ljust(5)} {s3.ljust(5)} {n}", end="")

            scale = 1.0
            if "ln_" in n or ".ln" in n or "time_" in n or "_mask" in n or "pos_emb" in n or '.mask.' in n or n.endswith('_w') or n.endswith('_w1') or n.endswith('_w2') or n.endswith('_bias') or (".weight" not in n):
                if 'ln_x.weight' in n:
                    layer_scale = (1+int(n.split('.')[1])) / self.config.n_layer
                    m[n] = (p * 0.0) + (layer_scale ** 0.7)
                else:
                    m[n] = p
                print()
            elif n == "emb.weight":
                m[n] = p
                scale = -1e-4
                nn.init.uniform_(m[n], a=scale, b=-scale)
                print(f" [scale {scale}]")
            elif n == "head.weight":
                m[n] = p
                if self.config.vocab_size > self.config.n_embd:
                    scale = 0.5 * math.sqrt(self.config.vocab_size / self.config.n_embd)
                else:
                    scale = 0.5
                nn.init.orthogonal_(m[n], gain=scale)
                print(f" [scale {scale}]")
            else:
                assert n.endswith('.weight') # should always be true

                zero = [".att.output.", ".ffn.value.", ".ffn.receptance.", ".ffnPre.value.", ".ffnPre.receptance.", "head_q.", '.oo.', '.rr.']

                for kk in zero:
                    if kk in n:
                        scale = 0

                for kk in [".att.key."]:
                    if kk in n:
                        scale = 0.1
                for kk in [".att.gate."]:
                    if kk in n:
                        scale = 0.1

                print(f" [scale {scale}]")

                if self.config.accelerator.upper() == "GPU":
                    m[n] = torch.empty((shape[0], shape[1]), device="cuda")
                else:
                    m[n] = torch.empty((shape[0], shape[1]))

                if scale == 0:
                    nn.init.zeros_(m[n])
                elif scale < 0:
                    nn.init.uniform_(m[n], a=scale, b=-scale)
                else:
                    nn.init.orthogonal_(m[n], gain=scale)

            m[n] = m[n].cpu()
            if os.environ.get("RWKV_FLOAT_MODE") == "fp16":
                m[n] = m[n].half()
            elif os.environ.get("RWKV_FLOAT_MODE") == "bf16":
                m[n] = m[n].bfloat16()
            n_params += m[n].numel()

        print('model params', n_params)
        gc.collect()
        torch.cuda.empty_cache()
        return m

    # ==================== RunningWay Inference Methods ====================
    def generate_with_state(self, input_tokens, max_length=100, temperature=1.0, keep_state=True):
        """
        Generate text using state pool for consistent long-context generation
        """
        self.eval()
        generated = input_tokens.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Get model output
                logits = self(generated)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop if we exceed context length
                if generated.size(1) >= self.config.ctx_len:
                    break
        
        # Reset state if not keeping it
        if not keep_state:
            self.reset_state()
            
        return generated
    
    def get_state_info(self):
        """Get information about current state usage"""
        if not self.using_state_pool or self.state_pool is None:
            return {"state_pool_enabled": False}
        
        info = {
            "state_pool_enabled": True,
            "has_system_prompt": self.state_pool.has_system_prompt,
            "allocation_ratios": {
                "system": self.state_pool.alpha_sys,
                "rnn": self.state_pool.alpha_rnn,
                "window": self.state_pool.alpha_win
            },
            "rnn_state_norm": self.state_pool.rnn_state.norm().item(),
            "window_state_norm": self.state_pool.window_state.norm().item()
        }
        
        if self.state_pool.has_system_prompt:
            info["system_state_norm"] = self.state_pool.system_state_buffer.norm().item()
            
        return info
    
    def print_model_info(self):
        """打印模型信息"""
        config = self.config
        print("=" * 60)
        print("RunningWay Model Information")
        print("=" * 60)
        print(f"Architecture:")
        print(f"   - Layers: {config.n_layer}")
        print(f"   - Embedding Dim: {config.n_embd}")
        print(f"   - Attention Dim: {config.dim_att}")
        print(f"   - Head Size: {config.head_size}")
        print(f"   - Context Length: {config.ctx_len}")
        
        print(f"\nMulti-State:")
        print(f"   - Enabled: {'√' if self.using_state_pool else '×'}")
        if self.using_state_pool:
            print(f"   - Window Size: {config.window_size}")
            print(f"   - Default Ratios: {config.default_state_ratios}")
            print(f"   - Reset Per Batch: {'√' if config.reset_state_per_batch else '×'}")
        
        print(f"\nTraining:")
        print(f"   - Learning Rate: {config.lr_init}")
        print(f"   - Weight Decay: {config.weight_decay}")
        print(f"   - Gradient Checkpoint: {'√' if config.grad_cp == 1 else '×'}")
        print(f"   - Load Model: {config.load_model}")
        print("=" * 60)
    # ==================== End RunningWay Inference Methods ====================
