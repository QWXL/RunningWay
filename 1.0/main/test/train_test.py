# train_runningway.py
import argparse
from pathlib import Path
from src.runningway.config import RunningWayConfig
from src.runningway.model import RunningWay  # 假设上述代码在 model.py 中

def parse_args():
    parser = argparse.ArgumentParser()
    
    # 基础模型参数
    parser.add_argument('--n_embd', type=int, default=768)
    parser.add_argument('--n_layer', type=int, default=12)
    parser.add_argument('--vocab_size', type=int, default=50277)
    parser.add_argument('--ctx_len', type=int, default=4096)
    parser.add_argument('--head_size', type=int, default=64)
    
    # RunningWay 特有参数
    parser.add_argument('--use_multi_state', action='store_true', default=True)
    parser.add_argument('--window_size', type=int, default=1024)
    parser.add_argument('--reset_state_per_batch', action='store_true', default=True)
    
    # 训练参数
    parser.add_argument('--lr_init', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--grad_cp', type=int, default=0)
    parser.add_argument('--my_testing', action='store_true', default=False)
    
    # 配置管理
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--save_config', type=str, help='保存配置到文件')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 创建配置
    if args.config and Path(args.config).exists():
        print(f"[Main] Loading config from {args.config}")
        config = RunningWayConfig.load(args.config)
        # 命令行参数覆盖配置文件
        config = config.merge_with(RunningWayConfig.from_args(args))
    else:
        print("[Main] Creating config from command line arguments")
        config = RunningWayConfig.from_args(args)
    
    # 从环境变量更新配置
    config.update_from_env()
    
    # 打印配置
    config.print_config()
    
    # 保存配置
    if args.save_config:
        config.save(args.save_config)
    
    # 创建模型
    print(f"[Main] Creating RunningWay model...")
    model = RunningWay(config)
    
    # 打印模型信息
    model.print_model_info()
    
    print(f"[Main] Model ready for training!")
    print(f"[Main] Total parameters: {sum(p.numel() for p in model.parameters())}")

if __name__ == "__main__":
    main()
