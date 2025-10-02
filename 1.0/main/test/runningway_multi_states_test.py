# runningway_test.py
import torch

def test_compatibility():
    """测试兼容性方案"""
    print("🧪 Testing RunningWay Compatibility...")
    
    # 测试数据
    B, T, C = 2, 32, 512
    
    # 创建测试张量
    q = torch.randn(B, T, C, dtype=torch.bfloat16).cuda()
    w = torch.randn(B, T, C, dtype=torch.bfloat16).cuda()
    k = torch.randn(B, T, C, dtype=torch.bfloat16).cuda()
    v = torch.randn(B, T, C, dtype=torch.bfloat16).cuda()
    a = torch.randn(B, T, C, dtype=torch.bfloat16).cuda()
    b = torch.randn(B, T, C, dtype=torch.bfloat16).cuda()
    
    # 测试原始函数
    output_original = RUN_CUDA_RWKV7g_original(q, w, k, v, a, b)
    print("✅ Original function works")
    
    # 测试禁用多状态
    from runningway_config import config
    config.disable_multi_state()
    
    output_disabled = RUN_CUDA_RWKV7g_compatible(q, w, k, v, a, b)
    torch.testing.assert_close(output_original, output_disabled)
    print("✅ Disabled multi-state matches original")
    
    # 测试启用多状态（回退模式）
    config.use_multi_state = True
    fused_state = torch.randn(B, C, dtype=torch.bfloat16).cuda()
    
    try:
        output_with_state = RUN_CUDA_RWKV7g_compatible(
            q, w, k, v, a, b, fused_state
        )
        print("✅ Multi-state fallback works")
        assert output_with_state.shape == (B, T, C)
        print("✅ Output shape correct")
    except Exception as e:
        print(f"❌ Multi-state fallback failed: {e}")
        return False
    
    print("🎉 All compatibility tests passed!")
    return True

def benchmark_performance():
    """性能基准测试"""
    import time
    
    B, T, C = 4, 1024, 512
    iterations = 100
    
    # 准备测试数据
    q = torch.randn(B, T, C, dtype=torch.bfloat16).cuda()
    w = torch.randn(B, T, C, dtype=torch.bfloat16).cuda()
    k = torch.randn(B, T, C, dtype=torch.bfloat16).cuda()
    v = torch.randn(B, T, C, dtype=torch.bfloat16).cuda()
    a = torch.randn(B, T, C, dtype=torch.bfloat16).cuda()
    b = torch.randn(B, T, C, dtype=torch.bfloat16).cuda()
    
    # 测试原始性能
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iterations):
        _ = RUN_CUDA_RWKV7g_original(q, w, k, v, a, b)
    torch.cuda.synchronize()
    original_time = time.time() - start
    
    # 测试多状态回退性能
    fused_state = torch.randn(B, C, dtype=torch.bfloat16).cuda()
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iterations):
        _ = RUN_CUDA_RWKV7g_compatible(q, w, k, v, a, b, fused_state)
    torch.cuda.synchronize()
    multi_state_time = time.time() - start
    
    print(f"📊 Performance Benchmark:")
    print(f"  Original: {original_time:.4f}s ({iterations} iterations)")
    print(f"  Multi-state (fallback): {multi_state_time:.4f}s ({iterations} iterations)")
    print(f"  Overhead: {((multi_state_time - original_time) / original_time * 100):.2f}%")
    
    return original_time, multi_state_time