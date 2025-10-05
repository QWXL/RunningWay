import os, math, time, datetime, subprocess
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only

def my_save(args, trainer, dd, ff):
    if 'deepspeed_stage_3' in args.strategy:
        trainer.save_checkpoint(ff, weights_only=True)
    else:
        torch.save(dd, ff)

class train_callback(pl.Callback):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        args = self.args
        
        # === RunningWay: 获取配置 ===
        if hasattr(pl_module, 'config'):
            config = pl_module.config
        else:
            # 回退到 args（兼容性）
            config = args

        real_step = trainer.global_step + args.epoch_begin * args.epoch_steps

        # LR schedule
        w_step = args.warmup_steps

        if args.my_exit_tokens != 0: # cosine decay
            real_tokens = real_step * config.ctx_len * args.real_bsz
            warmup_tokens = w_step * config.ctx_len * args.real_bsz
            progress = (real_tokens - warmup_tokens) / (abs(args.my_exit_tokens) - warmup_tokens)
            progress = max(0, min(1, progress))
            lr_final_factor = config.lr_final / config.lr_init                
            lr_mult = (0.5 + lr_final_factor / 2) + (0.5 - lr_final_factor / 2) * math.cos(math.pi * progress)
            if args.my_exit_tokens > 0:
                lr = config.lr_init * lr_mult
            else:
                lr = (lr + config.lr_init * lr_mult) / 2
            if progress >= 1:
                if (trainer.is_global_zero) or ('deepspeed_stage_3' in args.strategy):
                    my_save(
                        args, trainer,
                        pl_module.state_dict(),
                        f"{args.proj_dir}/runningway-final.pth",
                    )
                    exit(0)
        if trainer.global_step < w_step:
            lr = lr * (0.01 + 0.99 * trainer.global_step / w_step)

        wd_now = config.weight_decay

        for param_group in trainer.optimizers[0].param_groups:
            if param_group["weight_decay"] > 0:
                param_group["weight_decay"] = wd_now

        trainer.my_lr = lr
        trainer.my_wd = wd_now

        if trainer.global_step == 0:
            if trainer.is_global_zero:  # logging
                trainer.my_loss_sum = 0
                trainer.my_loss_count = 0
                trainer.my_log = open(args.proj_dir + "/train_log.txt", "a")
                trainer.my_log.write(f"NEW RUN {args.my_timestamp}\n{vars(self.args)}\n")
                
                # === RunningWay: 记录配置信息 ===
                if hasattr(pl_module, 'config'):
                    trainer.my_log.write(f"\nRunningWay Configuration:\n")
                    trainer.my_log.write(f"Multi-State: {'Enabled' if config.use_multi_state else 'Disabled'}\n")
                    if config.use_multi_state:
                        trainer.my_log.write(f"Window Size: {config.window_size}\n")
                        trainer.my_log.write(f"State Ratios: {config.default_state_ratios}\n")
                        trainer.my_log.write(f"Reset Per Batch: {config.reset_state_per_batch}\n")
                    trainer.my_log.write(f"Learning Rate: {config.lr_init} to {config.lr_final}\n")
                    trainer.my_log.write(f"Weight Decay: {config.weight_decay}\n")
                    trainer.my_log.write("\n")
                
                try:
                    print(f"\n{trainer.strategy.config}\n")
                    trainer.my_log.write(f"{trainer.strategy.config}\n")
                except:
                    pass
                trainer.my_log.flush()
                if len(args.wandb) > 0:
                    print("Login to wandb...")
                    import wandb
                    
                    # === RunningWay: 准备 wandb 配置 ===
                    wandb_config = vars(args)
                    if hasattr(pl_module, 'config'):
                        wandb_config.update({
                            'runningway_multi_state': config.use_multi_state,
                            'runningway_window_size': config.window_size,
                            'runningway_state_ratios': config.default_state_ratios,
                            'runningway_system_frozen': config.system_prompt_frozen,
                        })
                    
                    wandb.init(
                        project=args.wandb,
                        name=args.run_name + " " + args.my_timestamp,
                        config=wandb_config,
                        save_code=False,
                    )
                    trainer.my_wandb = wandb

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        args = self.args
        
        # === RunningWay: 获取配置 ===
        if hasattr(pl_module, 'config'):
            config = pl_module.config
        else:
            config = args
        
        token_per_step = config.ctx_len * args.real_bsz
        real_step = trainer.global_step + args.epoch_begin * args.epoch_steps

        if trainer.is_global_zero:  # logging
            t_now = time.time_ns()
            kt_s = 0
            try:
                t_cost = (t_now - trainer.my_time_ns) / 1e9
                kt_s = token_per_step / t_cost / 1000
                self.log("REAL it/s", 1.0 / t_cost, prog_bar=True, on_step=True)
                self.log("Kt/s", kt_s, prog_bar=True, on_step=True)
            except:
                pass
            trainer.my_time_ns = t_now
            trainer.my_loss = trainer.my_loss_all.float().mean().item()
            trainer.my_loss_sum += trainer.my_loss
            trainer.my_loss_count += 1
            trainer.my_epoch_loss = trainer.my_loss_sum / trainer.my_loss_count
            self.log("lr", trainer.my_lr, prog_bar=True, on_step=True)
            self.log("loss", trainer.my_epoch_loss, prog_bar=True, on_step=True)

            # === RunningWay: 状态监控日志 ===
            if hasattr(pl_module, 'get_state_info') and real_step % 100 == 0:  # 每100步记录一次状态信息
                try:
                    state_info = pl_module.get_state_info()
                    if state_info.get("state_pool_enabled", False):
                        self.log("state_rnn_norm", state_info.get("rnn_state_norm", 0), on_step=True)
                        self.log("state_window_norm", state_info.get("window_state_norm", 0), on_step=True)
                        if "system_state_norm" in state_info:
                            self.log("state_system_norm", state_info["system_state_norm"], on_step=True)
                        
                        # 记录状态分配比例
                        if "allocation_ratios" in state_info:
                            for key, value in state_info["allocation_ratios"].items():
                                self.log(f"state_ratio_{key}", value, on_step=True)
                except Exception as e:
                    print(f"Error logging state info: {e}")

            if len(args.wandb) > 0:
                lll = {
                    "loss": trainer.my_loss, 
                    "lr": trainer.my_lr, 
                    "wd": trainer.my_wd, 
                    "Gtokens": real_step * token_per_step / 1e9
                }
                if kt_s > 0:
                    lll["kt/s"] = kt_s
                
                # === RunningWay: wandb 状态信息 ===
                if hasattr(pl_module, 'get_state_info'):
                    try:
                        state_info = pl_module.get_state_info()
                        if state_info.get("state_pool_enabled", False):
                            lll.update({
                                "state_rnn_norm": state_info.get("rnn_state_norm", 0),
                                "state_window_norm": state_info.get("window_state_norm", 0),
                            })
                            if "system_state_norm" in state_info:
                                lll["state_system_norm"] = state_info["system_state_norm"]
                            
                            if "allocation_ratios" in state_info:
                                for key, value in state_info["allocation_ratios"].items():
                                    lll[f"state_ratio_{key}"] = value
                    except Exception as e:
                        print(f"Error adding state info to wandb: {e}")
                
                trainer.my_wandb.log(lll, step=int(real_step))

        if (trainer.is_global_zero) or ('deepspeed_stage_3' in args.strategy): # save pth
            if args.magic_prime > 0:
                if int(real_step) == int(args.magic_prime // args.real_bsz) - 1:
                    to_save_dict = pl_module.state_dict()
                    my_save(
                        args, trainer,
                        to_save_dict,
                        f"{args.proj_dir}/runningway-final.pth",
                    )

    def on_train_epoch_start(self, trainer, pl_module):
        args = self.args
        dataset = trainer.train_dataloader.dataset.datasets
        assert "MyDataset" in str(dataset)
        dataset.global_rank = trainer.global_rank
        dataset.real_epoch = int(args.epoch_begin + trainer.current_epoch)
        dataset.world_size = trainer.world_size
        
        # === RunningWay: Epoch 开始时的状态日志 ===
        if trainer.is_global_zero and hasattr(pl_module, 'get_state_info'):
            try:
                state_info = pl_module.get_state_info()
                rank_zero_info(f"Epoch {dataset.real_epoch} State Info: {state_info}")
            except Exception as e:
                print(f"Error getting state info at epoch start: {e}")

    def on_train_epoch_end(self, trainer, pl_module):
        args = self.args
        to_save_dict = {}
        if (trainer.is_global_zero) or ('deepspeed_stage_3' in args.strategy):  # save pth
            if (args.epoch_save > 0 and trainer.current_epoch % args.epoch_save == 0) or (trainer.current_epoch == args.epoch_count - 1):
                if args.data_type == 'wds_img':
                    raw_dict = pl_module.state_dict()
                    for k in raw_dict:
                        if k.startswith('encoder.') or k.startswith('decoder.'):
                            to_save_dict[k] = raw_dict[k]
                else:
                    to_save_dict = pl_module.state_dict()
                try:
                    # === RunningWay: 保存文件名更新 ===
                    save_name = f"{args.proj_dir}/runningway-{args.epoch_begin + trainer.current_epoch}.pth"
                    
                    # 同时保存配置文件
                    if hasattr(pl_module, 'config'):
                        config_save_name = f"{args.proj_dir}/runningway-{args.epoch_begin + trainer.current_epoch}-config.json"
                        pl_module.config.save(config_save_name)
                        rank_zero_info(f"Configuration saved to: {config_save_name}")
                    
                    my_save(args, trainer, to_save_dict, save_name)
                    rank_zero_info(f"Model saved to: {save_name}")
                    
                except Exception as e:
                    print('Error\n\n', e, '\n\n')

        if trainer.is_global_zero:  # logging
            # === RunningWay: 增强的日志信息 ===
            log_line = f"{args.epoch_begin + trainer.current_epoch} {trainer.my_epoch_loss:.6f} {math.exp(trainer.my_epoch_loss):.4f} {trainer.my_lr:.8f} {datetime.datetime.now()} {trainer.current_epoch}"
            
            # 添加状态信息到日志
            if hasattr(pl_module, 'get_state_info'):
                try:
                    state_info = pl_module.get_state_info()
                    if state_info.get("state_pool_enabled", False):
                        ratios = state_info.get("allocation_ratios", {})
                        log_line += f" state_ratios=({ratios.get('system', 0):.2f},{ratios.get('rnn', 0):.2f},{ratios.get('window', 0):.2f})"
                except:
                    pass
            
            trainer.my_log.write(log_line + "\n")
            trainer.my_log.flush()

            trainer.my_loss_sum = 0
            trainer.my_loss_count = 0

@rank_zero_only
def generate_init_weight(model, init_weight_name):
    mm = model.generate_init_weight()

    # === RunningWay: 使用 config 而不是 args ===
    if hasattr(model, 'config'):
        config = model.config
    else:
        config = model.args

    if config.train_stage == 1:
        load_model_is_valid = False
        
        # Check if load_model file exists
        if len(config.load_model) > 0 and config.load_model != "0" and os.path.isfile(config.load_model):
            load_model_is_valid = True
        
        if load_model_is_valid:
            print(f"Combine weights from {config.load_model}...")
            load_dict = torch.load(config.load_model, map_location="cpu")
            for k in load_dict:
                try:
                    assert k in mm
                except:
                    print('missing', k)
                    exit(0)
                src = load_dict[k]
                try:
                    mm[k] = src.reshape(mm[k].shape)
                except:
                    tmp = mm[k].squeeze().clone()
                    print(k, src.shape, '-->', mm[k].shape)
                    ss = src.shape[0]
                    dd = tmp.shape[0]
                    for i in range(dd):
                        pos = i / dd * ss
                        if pos >= ss - 1:
                            tmp[i] = src[ss-1]
                        else:
                            p0 = int(math.floor(pos))
                            ii = pos - p0
                            tmp[i] = src[p0] * (1-ii) + src[p0+1] * (ii)
                    mm[k] = tmp.reshape(mm[k].shape)
                    sss = src.squeeze().float().cpu().numpy()
                    print(sss[:10], '...', sss[-10:])
                    mmm = mm[k].squeeze().float().cpu().numpy()
                    print(mmm[:10], '...', mmm[-10:])

    print(f"Save to {init_weight_name}...")
    torch.save(mm, init_weight_name)

    if config.train_stage == 1:
        print("Done. Now go for stage 2.")
        exit(0)
