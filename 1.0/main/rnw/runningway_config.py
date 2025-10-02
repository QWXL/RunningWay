# src/model_runningway/config.py
import os
import json
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

@dataclass
class RunningWayConfig:
    """RunningWay 模型配置管理类"""
    
    # === 基础模型参数 ===
    vocab_size: int = 50277
    n_embd: int = 768
    n_layer: int = 12
    ctx_len: int = 4096
    
    # === 注意力参数 ===
    dim_att: Optional[int] = None  # None 表示使用 n_embd
    head_size: int = 64
    
    # === FFN 参数 ===
    dim_ffn: Optional[int] = None  # None 表示自动计算
    
    # === RunningWay 特有参数 ===
    # 多状态相关
    use_multi_state: bool = True
    window_size: int = 1024
    default_state_ratios: Dict[str, float] = field(default_factory=lambda: {
        'system': 0.3,
        'window': 0.4, 
        'rnn': 0.3
    })
    
    # 系统提示相关
    system_prompt_frozen: bool = True
    system_recall_threshold: float = 0.5
    
    # 训练相关
    reset_state_per_batch: bool = True
    state_pool_learning_rate: float = 1e-4
    
    # 兼容性标志
    use_new_cuda_kernel: bool = False
    fallback_to_python: bool = False
    
    # === 训练参数 ===
    lr_init: float = 1e-4
    betas: tuple = (0.9, 0.999)
    adam_eps: float = 1e-8
    weight_decay: float = 0.01
    grad_cp: int = 0  # 梯度检查点
    
    # === 其他参数 ===
    my_testing: bool = False
    accelerator: str = "gpu"
    
    def __post_init__(self):
        """初始化后处理"""
        # 自动计算 dim_att
        if self.dim_att is None:
            self.dim_att = self.n_embd
            
        # 自动计算 dim_ffn
        if self.dim_ffn is None:
            self.dim_ffn = int((self.n_embd * 3.5) // 32 * 32)
            
        # 验证参数
        assert self.n_embd % 32 == 0
        assert self.dim_att % 32 == 0
        assert self.dim_ffn % 32 == 0
    
    @classmethod
    def from_args(cls, args) -> 'RunningWayConfig':
        """从 argparse.Namespace 创建配置"""
        config_dict = {}
        
        # 复制所有属性
        for key, value in vars(args).items():
            if hasattr(cls, key):
                config_dict[key] = value
        
        return cls(**config_dict)
    
    def to_args(self):
        """转换为 argparse.Namespace"""
        import argparse
        args = argparse.Namespace()
        
        for key, value in self.__dict__.items():
            setattr(args, key, value)
            
        return args
    
    def update_from_env(self):
        """从环境变量更新配置"""
        env_mappings = {
            'RUNNINGWAY_USE_MULTI_STATE': ('use_multi_state', lambda x: x.lower() == 'true'),
            'RUNNINGWAY_WINDOW_SIZE': ('window_size', int),
            'RUNNINGWAY_USE_NEW_CUDA': ('use_new_cuda_kernel', lambda x: x.lower() == 'true'),
            'RUNNINGWAY_PYTHON_FALLBACK': ('fallback_to_python', lambda x: x.lower() == 'true'),
            'RUNNINGWAY_CTX_LEN': ('ctx_len', int),
            'RUNNINGWAY_LR_INIT': ('lr_init', float),
        }
        
        for env_var, (config_key, converter) in env_mappings.items():
            if env_var in os.environ:
                try:
                    setattr(self, config_key, converter(os.environ[env_var]))
                    print(f"[Config] Updated {config_key} from environment: {getattr(self, config_key)}")
                except (ValueError, TypeError) as e:
                    print(f"[Config] Failed to parse {env_var}: {e}")
    
    def save(self, path: str):
        """保存配置到文件"""
        config_dict = self.__dict__.copy()
        
        # 处理不可序列化的对象
        if 'default_state_ratios' in config_dict:
            config_dict['default_state_ratios'] = dict(config_dict['default_state_ratios'])
            
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2)
        print(f"[Config] Saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'RunningWayConfig':
        """从文件加载配置"""
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        return cls(**config_dict)
    
    def print_config(self):
        """打印配置信息"""
        print("=" * 60)
        print("RunningWay Configuration")
        print("=" * 60)
        
        print(f"📊 Model Size: {self.n_layer} layers, {self.n_embd} dim, {self.ctx_len} ctx_len")
        print(f"⚙️  Multi-State: {'Enabled' if self.use_multi_state else 'Disabled'}")
        if self.use_multi_state:
            print(f"   - Window Size: {self.window_size}")
            print(f"   - State Ratios: {self.default_state_ratios}")
            print(f"   - System Prompt Frozen: {self.system_prompt_frozen}")
        print(f"🔧 CUDA Kernel: {'New' if self.use_new_cuda_kernel else 'Original + Fallback'}")
        print(f"📈 Training: LR={self.lr_init}, Weight Decay={self.weight_decay}")
        print("=" * 60)
