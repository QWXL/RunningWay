#!/usr/bin/env python3
########################################################################################################
# RunningWay Stage 1 Training Script - Enhanced RWKV with Multi-State Architecture
# Github: https://github.com/QWXL/RunningWay
# Based on RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import logging
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    from argparse import ArgumentParser
    from pytorch_lightning import Trainer
    from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
    import pytorch_lightning as pl

    rank_zero_info("########## RunningWay Stage 1: Multi-State Architecture ##########")

    parser = ArgumentParser()

    # === 基础模型参数 ===
    parser.add_argument("--load_model", default="", type=str)  # full path, with .pth
    parser.add_argument("--wandb", default="", type=str)  # wandb project name. if "" then don't use wandb
    parser.add_argument("--proj_dir", default="out", type=str)
    parser.add_argument("--random_seed", default="-1", type=int)
    parser.add_argument("--experiment_name", default="runningway_stage1", type=str)

    parser.add_argument("--data_file", default="", type=str)
    parser.add_argument("--data_type", default="utf-8", type=str)
    parser.add_argument("--vocab_size", default=0, type=int)  # vocab_size = 0 means auto

    parser.add_argument("--ctx_len", default=1024, type=int)
    parser.add_argument("--epoch_steps", default=1000, type=int)
    parser.add_argument("--epoch_count", default=500, type=int)
    parser.add_argument("--epoch_begin", default=0, type=int)
    parser.add_argument("--epoch_save", default=5, type=int)

    parser.add_argument("--micro_bsz", default=12, type=int)
    parser.add_argument("--n_layer", default=6, type=int)
    parser.add_argument("--n_embd", default=512, type=int)
    parser.add_argument("--dim_att", default=0, type=int)
    parser.add_argument("--dim_ffn", default=0, type=int)

    # === 训练参数 ===
    parser.add_argument("--lr_init", default=6e-4, type=float)
    parser.add_argument("--lr_final", default=1e-5, type=float)
    parser.add_argument("--warmup_steps", default=-1, type=int)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.99, type=float)
    parser.add_argument("--adam_eps", default=1e-18, type=float)
    parser.add_argument("--grad_cp", default=0, type=int)
    parser.add_argument("--weight_decay", default=0, type=float)
    parser.add_argument("--grad_clip", default=1.0, type=float)

    # === RunningWay 特有参数 ===
    parser.add_argument("--use_multi_state", default=True, type=lambda x: (str(x).lower() == 'true'), help="Enable multi-state mechanism")
    parser.add_argument("--window_size", type=int, default=1024, help="Sliding window size for state")
    parser.add_argument("--reset_state_per_batch", default=True, type=lambda x: (str(x).lower() == 'true'), help="Reset state for each training batch")
    parser.add_argument("--system_prompt_frozen", default=True, type=lambda x: (str(x).lower() == 'true'), help="Freeze system prompt state")
    parser.add_argument("--state_pool_learning_rate", type=float, default=1e-4, help="Learning rate for state pool parameters")
    
    # === 状态分配比例 ===
    parser.add_argument("--system_ratio", type=float, default=0.3, help="System state ratio")
    parser.add_argument("--rnn_ratio", type=float, default=0.4, help="RNN state ratio")
    parser.add_argument("--window_ratio", type=float, default=0.3, help="Window state ratio")

    # === 兼容性参数 ===
    parser.add_argument("--use_new_cuda_kernel", default=False, type=lambda x: (str(x).lower() == 'true'), help="Use new CUDA kernel for multi-state")
    parser.add_argument("--fallback_to_python", default=False, type=lambda x: (str(x).lower() == 'true'), help="Fallback to Python implementation")

    # === 原始 RWKV 参数 ===
    parser.add_argument("--train_stage", default=0, type=int)
    parser.add_argument("--ds_bucket_mb", default=200, type=int)
    parser.add_argument("--head_size", default=64, type=int)
    parser.add_argument("--load_partial", default=0, type=int)
    parser.add_argument("--magic_prime", default=0, type=int)
    parser.add_argument("--my_testing", default='r010', type=str)
    parser.add_argument("--my_exit_tokens", default=0, type=int)

    # === 配置管理 ===
    parser.add_argument("--config", type=str, help="Path to RunningWayConfig file")
    parser.add_argument("--save_config", type=str, help="Save configuration to file")

    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    ########################################################################################################

    import os, warnings, math, datetime, sys, time
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    if "deepspeed" in args.strategy:
        import deepspeed
    from pytorch_lightning import seed_everything

    # === 导入 RunningWay 配置和模型 ===
    from rnw.runningway_config import RunningWayConfig
    from src.model import RunningWay

    if args.random_seed >= 0:
        print(f"########## WARNING: GLOBAL SEED {args.random_seed} THIS WILL AFFECT MULTIGPU SAMPLING ##########\n" * 3)
        seed_everything(args.random_seed)

    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")
    warnings.filterwarnings("ignore", ".*The progress bar already tracks a metric with the*")

    ########################################################################################################
    # === 创建 RunningWay 配置 ===
    ########################################################################################################
    
    rank_zero_info("Creating RunningWayConfig from command line arguments")
    # 创建临时 args 对象
    temp_args = type('TempArgs', (), {})()
    for key, value in vars(args).items():
        if hasattr(RunningWayConfig, key):
            setattr(temp_args, key, value)
    config = RunningWayConfig.from_args(temp_args)
    config.post_init()
    # 设置 RunningWay 特有参数
    config.use_multi_state = args.use_multi_state
    config.window_size = args.window_size
    config.reset_state_per_batch = args.reset_state_per_batch
    config.system_prompt_frozen = args.system_prompt_frozen
    config.state_pool_learning_rate = args.state_pool_learning_rate
    config.use_new_cuda_kernel = args.use_new_cuda_kernel
    config.fallback_to_python = args.fallback_to_python
    
    # 设置状态分配比例
    config.default_state_ratios = {
        'system': args.system_ratio,
        'rnn': args.rnn_ratio,
        'window': args.window_ratio
    }
    
    # 从环境变量更新配置
    config.update_from_env()
    
    # 保存配置
    if args.save_config:
        config.save(args.save_config)
    
    # 打印配置
    config.print_config()
    
    ########################################################################################################
    # === 设置训练参数 ===
    ########################################################################################################

    args.my_timestamp = datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
    args.enable_checkpointing = False
    args.replace_sampler_ddp = False
    args.logger = False
    args.gradient_clip_val = args.grad_clip
    args.num_sanity_val_steps = 0
    args.check_val_every_n_epoch = int(1e20)
    args.log_every_n_steps = int(1e20)
    args.max_epochs = -1  # continue forever
    args.betas = (args.beta1, args.beta2)
    args.real_bsz = int(args.num_nodes) * int(args.devices) * args.micro_bsz
    os.environ["RNW_MY_TESTING"] = args.my_testing
    os.environ["RNW_CTXLEN"] = str(config.ctx_len)
    os.environ["RNW_HEAD_SIZE"] = str(config.head_size)
    
    # 设置环境变量
    if config.use_multi_state:
        os.environ["RUNNINGWAY_USE_MULTI_STATE"] = "1"
    if config.use_new_cuda_kernel:
        os.environ["RUNNINGWAY_USE_NEW_CUDA"] = "1"
    if config.fallback_to_python:
        os.environ["RUNNINGWAY_PYTHON_FALLBACK"] = "1"
    os.environ["RUNNINGWAY_WINDOW_SIZE"] = str(config.window_size)

    # 自动计算维度
    if args.dim_att <= 0:
        args.dim_att = config.n_embd
    if args.dim_ffn <= 0:
        args.dim_ffn = int((config.n_embd * 3.5) // 32 * 32)

    # 设置运行名称
    if config.use_multi_state:
        args.run_name = f"RunningWay-{config.n_layer}L-{config.n_embd}D-multiState"
    else:
        args.run_name = f"RunningWay-{config.n_layer}L-{config.n_embd}D-standard"
    
    if not os.path.exists(args.proj_dir):
        os.makedirs(args.proj_dir)

    # 计算训练参数
    if args.magic_prime > 0:
        args.epoch_count = args.magic_prime // 40320
        args.epoch_steps = 40320 // args.real_bsz
        assert args.epoch_steps * args.real_bsz == 40320

    samples_per_epoch = args.epoch_steps * args.real_bsz
    tokens_per_epoch = samples_per_epoch * config.ctx_len
    
    try:
        deepspeed_version = deepspeed.__version__
    except:
        deepspeed_version = None
        pass
    
    rank_zero_info(
        f"""
############################################################################
#
# RunningWay Stage 1 {args.precision.upper()} on {args.num_nodes}x{args.devices} {args.accelerator.upper()}
# Multi-State: {'Enabled' if config.use_multi_state else 'Disabled'}
# Window Size: {config.window_size if config.use_multi_state else 'N/A'}
# State Ratios: {config.default_state_ratios if config.use_multi_state else 'N/A'}
#
# Data = {args.data_file} ({args.data_type}), ProjDir = {args.proj_dir}
#
# Epoch = {args.epoch_begin} to {args.epoch_begin + args.epoch_count - 1}, save every {args.epoch_save} epoch
#
# Each "epoch" = {args.epoch_steps} steps, {samples_per_epoch} samples, {tokens_per_epoch} tokens
#
# Model = {config.n_layer} n_layer, {config.n_embd} n_embd, {config.ctx_len} ctx_len
#
# Adam = lr {config.lr_init} to {config.lr_final}, warmup {args.warmup_steps} steps, beta {config.betas}, eps {config.adam_eps}
#
# Found torch {torch.__version__}, recommend latest torch
# Found deepspeed {deepspeed_version}, recommend latest deepspeed
# Found pytorch_lightning {pl.__version__}, recommend 1.9.5
#
############################################################################
"""
    )
    rank_zero_info(str(vars(args)) + "\n")
    rank_zero_info("RunningWay Config:\n" + str(config.__dict__) + "\n")

    assert args.data_type in ["binidx", "utf-8"]

    if config.lr_final == 0 or config.lr_init == 0:
        rank_zero_info("\n\nNote: lr_final = 0 or lr_init = 0. Using linear LR schedule instead.\n\n")

    assert args.precision in ["fp32", "tf32", "fp16", "bf16"]
    os.environ["RWKV_FLOAT_MODE"] = args.precision
    if args.precision == "fp32":
        for i in range(10):
            rank_zero_info("\n\nNote: you are using fp32 (very slow). Try bf16 / tf32 for faster training.\n\n")
    if args.precision == "fp16":
        rank_zero_info("\n\nNote: you are using fp16 (might overflow). Try bf16 / tf32 for stable training.\n\n")

    os.environ["RWKV_JIT_ON"] = "1"
    if "deepspeed_stage_3" in args.strategy:
        os.environ["RWKV_JIT_ON"] = "0"  # somehow incompatible

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    if args.precision == "fp32":
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    else:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

    # PyTorch Lightning 1.9.5 AMP 配置
    if args.precision == "fp32" or args.precision == "tf32":
        args.precision = 32
    elif args.precision == "fp16":
        args.precision = 16
    elif args.precision == "bf16":
        args.precision = "bf16"


    ########################################################################################################

    from src.trainer import train_callback, generate_init_weight
    from src.dataset import MyDataset

    train_data = MyDataset(args)
    config.vocab_size = train_data.vocab_size

    # === 创建 RunningWay 模型 ===
    rank_zero_info(f"Creating RunningWay model with multi-state architecture...")
    model = RunningWay(config)
    
    # 打印模型信息
    model.print_model_info()

    # === 权重初始化和加载 ===
    if len(args.load_model) == 0 or args.train_stage == 1:
        init_weight_name = f"{args.proj_dir}/runningway-init.pth"
        rank_zero_info(f"Generating initial weights: {init_weight_name}")
        generate_init_weight(model, init_weight_name)
        args.load_model = init_weight_name

    rank_zero_info(f"########## Loading {args.load_model}... ##########")
    try:
        load_dict = torch.load(args.load_model, map_location="cpu")
        load_keys = list(load_dict.keys())
        for k in load_keys:
            if k.startswith('_forward_module.'):
                load_dict[k.replace('_forward_module.','')] = load_dict[k]
                del load_dict[k]
    except:
        rank_zero_info(f"Bad checkpoint {args.load_model}")
        if args.train_stage >= 2:
            max_p = getattr(args, 'my_pile_prev_p', None)
            if max_p is not None:
                if max_p == -1:
                    args.load_model = f"{args.proj_dir}/runningway-init.pth"
                else:
                    args.load_model = f"{args.proj_dir}/runningway-{max_p}.pth"
                args.epoch_begin = max_p + 1
                rank_zero_info(f"Trying {args.load_model}")
                load_dict = torch.load(args.load_model, map_location="cpu")

    if args.load_partial == 1:
        load_keys = load_dict.keys()
        for k in model.state_dict():
            if k not in load_keys:
                load_dict[k] = model.state_dict()[k]
    
    model.load_state_dict(load_dict)

    # === 设置训练器 ===
    trainer = Trainer.from_argparse_args(
        args,
        callbacks=[train_callback(args)],
    )

    if trainer.global_rank == 0:
        rank_zero_info("Model parameters:")
        for n in model.state_dict():
            shape = model.state_dict()[n].shape
            s0 = str(shape[0]) if len(shape) > 0 else ""
            s1 = str(shape[1]) if len(shape) > 1 else ""
            s2 = str(shape[2]) if len(shape) > 2 else ""
            s3 = str(shape[3]) if len(shape) > 3 else ""
            print(f"{s0.ljust(5)} {s1.ljust(5)} {s2.ljust(5)} {s3.ljust(5)} {n}")
        
        # 打印 RunningWay 特有信息
        rank_zero_info("\nRunningWay Stage 1 Features:")
        rank_zero_info(f"  - Multi-State: {'√' if config.use_multi_state else '×'}")
        if config.use_multi_state:
            rank_zero_info(f"  - Window Size: {config.window_size}")
            rank_zero_info(f"  - State Ratios: {config.default_state_ratios}")
            rank_zero_info(f"  - Reset Per Batch: {'√' if config.reset_state_per_batch else '×'}")
        rank_zero_info(f"  - CUDA Kernel: {'New' if config.use_new_cuda_kernel else 'Original + Fallback'}")

    if "deepspeed" in args.strategy:
        trainer.strategy.config["zero_optimization"]["allgather_bucket_size"] = args.ds_bucket_mb * 1000 * 1000
        trainer.strategy.config["zero_optimization"]["reduce_bucket_size"] = args.ds_bucket_mb * 1000 * 1000

    # must set shuffle=False, persistent_workers=False (because worker is in another thread)
    data_loader = DataLoader(
        train_data, 
        shuffle=False, 
        pin_memory=True, 
        batch_size=args.micro_bsz, 
        num_workers=1, 
        persistent_workers=False, 
        drop_last=True
    )

    if trainer.global_rank == 0:
        print(f'### Preparing for RunningWay Stage 1 training (loaded {args.load_model}). Please wait...')
    
    # === 开始训练 ===
    trainer.fit(model, data_loader)
    
    # === 训练完成后的信息 ===
    if trainer.global_rank == 0:
        rank_zero_info("\n" + "="*60)
        rank_zero_info("RunningWay Stage 1 Training Completed!")
        rank_zero_info("="*60)
        
        # 获取最终状态信息
        if config.use_multi_state:
            state_info = model.get_state_info()
            rank_zero_info(f"Final State Info: {state_info}")
        
        # 保存最终配置
        final_config_path = f"{args.proj_dir}/runningway-final-config.json"
        config.save(final_config_path)
        rank_zero_info(f"Final configuration saved to: {final_config_path}")