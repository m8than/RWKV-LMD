import os, math, time, datetime, subprocess
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from deepspeed.checkpoint.utils import clone_tensors_for_torch_save
import shutil

import re
from collections import OrderedDict
import torch.distributed as dist
from deepspeed.utils import safe_get_full_fp32_param
import glob
import math

def get_state_dict(module, return_params):
    # Dictionary to store parameters, initialized to None
    all_params = {}

    # Fill the dictionary with local parameters
    for name, param in module.named_parameters():
        full_param = safe_get_full_fp32_param(param)  # Full precision
        if return_params and name not in all_params.keys():
            all_params[name] = full_param.detach().cpu().clone().bfloat16()

    return all_params
    
# dd = state dict
# ff = filename
def my_save(args, trainer, dd, ff, model):
    if '14b-run1' in ff:
        fn = ff.split('/')[-1]
        fff = '/dev/shm/' + fn
        torch.save(dd, fff)
        subprocess.Popen(f" aws s3 mv {fff} s3://rwkv-14b-4k/{fn} --quiet", shell=True)
    elif ('world/14b' in ff) or ('world/7b' in ff):
        aa = ff.split('/')[1]
        fn = ff.split('/')[-1]
        fff = f'/dev/shm/{aa}-{fn}'
        torch.save(dd, fff)
        subprocess.Popen(f" aws s3 mv {fff} s3://rwkv-world/{aa}-{fn} --quiet", shell=True)
    else:
        if 'deepspeed_stage_3' in args.strategy:
            state_dict = get_state_dict(model, trainer.global_rank == 0)
            if trainer.global_rank == 0:
                torch.save(state_dict, ff)
        else:
            torch.save(dd, ff)

class train_callback(pl.Callback):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def on_train_start(self, trainer, pl_module):
        self.trainer = self.args.trainer
        
        total_devices = self.args.devices * self.args.num_nodes
        # self.args.dataset_len = document count
        #self.args.steps_per_epoch = self.args.dataset_len // (self.args.real_bsz)
        self.args.steps_per_epoch = trainer.estimated_stepping_batches
        if self.args.start_step > 0:
            global_step = self.args.start_step % self.args.steps_per_epoch
            epoch = self.args.start_step // self.args.steps_per_epoch
            docs_run = (global_step) * self.args.real_bsz
            
            self.set_current_epoch(epoch) # This is being loaded from the model
            self.set_total_batch_idx(docs_run)
            self.set_global_step(global_step) 

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        args = self.args
        # if args.cuda_cleanup > 0:
        #     torch.cuda.empty_cache()
        real_step = trainer.global_step
        lr_period = self.args.lr_step_period if self.args.lr_step_period != -1 else args.steps_per_epoch
        
        if hasattr(args, "this_run_steps"):
            args.this_run_steps += 1
        else:
            args.this_run_steps = 0

        # LR schedule
        w_step = args.warmup_steps
        if args.lr_final == args.lr_init or real_step == 0:
            lr = args.lr_init
        else:  # exp decay
            lr = args.lr_init * math.exp(math.log(args.lr_final / args.lr_init) * pow(progress, 1))
        # if trainer.is_global_zero:
        #     print(trainer.global_step, decay_step, decay_total, w_step, progress, lr)
        
        real_tokens = real_step * args.ctx_len * args.real_bsz
        warmup_tokens = w_step * args.ctx_len * args.real_bsz
        epoch_tokens = args.steps_per_epoch * args.ctx_len
        lr_period_tokens = args.lr_step_period if args.lr_step_period != -1 else args.steps_per_epoch
        progress = (real_tokens - warmup_tokens) / (abs(lr_period_tokens) - warmup_tokens)
        progress = max(0, min(1, progress))
        lr_final_factor = args.lr_final / args.lr_init                
        lr_mult = (0.5 + lr_final_factor / 2) + (0.5 - lr_final_factor / 2) * math.cos(math.pi * progress)


        if args.weight_decay_final > 0:
            wd_now = args.weight_decay * math.exp(math.log(args.weight_decay_final / args.weight_decay) * progress)
        else:
            wd_now = args.weight_decay

        for param_group in trainer.optimizers[0].param_groups:
            if param_group["weight_decay"] > 0:
                param_group["weight_decay"] = wd_now
            if args.layerwise_lr > 0:
                param_group["lr"] = lr * param_group["my_lr_scale"]
                # print(param_group["lr"], param_group["my_lr_scale"])
            else:
                param_group["lr"] = lr

        trainer.my_lr = lr
        trainer.my_wd = wd_now

        if trainer.is_global_zero and not hasattr(trainer, "my_loss_sum"):  # logging
            trainer.my_loss_sum = 0
            trainer.my_loss_count = 0
            trainer.my_log = open(args.proj_dir + "/train_log.txt", "a")
            trainer.my_log.write(f"NEW RUN {args.my_timestamp}\n{vars(self.args)}\n")
            try:
                print(f"\n{trainer.strategy.config}\n")
                trainer.my_log.write(f"{trainer.strategy.config}\n")
            except:
                pass
            trainer.my_log.flush()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        args = self.args
        token_per_step = args.ctx_len * args.real_bsz
        real_step = trainer.global_step
        if trainer.is_global_zero:  # logging
            t_now = time.time_ns()
            kt_s = 0
            try:
                t_cost = (t_now - trainer.my_time_ns) / 1e9
                kt_s = token_per_step / t_cost / 1000
                pl_module.log("REAL it/s", 1.0 / t_cost, prog_bar=True, on_step=True)
                pl_module.log("Kt/s", kt_s, prog_bar=True, on_step=True)
            except:
                pass
            trainer.my_time_ns = t_now
            trainer.my_loss = trainer.avg_loss
            trainer.my_loss_sum += trainer.my_loss
            trainer.my_loss_count += 1
            trainer.my_epoch_loss = trainer.my_loss_sum / trainer.my_loss_count
            self.log("lr", trainer.my_lr, prog_bar=True, on_step=True)
            self.log("loss", trainer.my_loss, prog_bar=True, on_step=True)

            
            
            # self.log("s", real_step, prog_bar=True, on_step=True)

            if len(args.wandb) > 0:
                lll = {"loss": trainer.my_loss, "lr": trainer.my_lr, "wd": trainer.my_wd, "Gtokens": real_step * token_per_step / 1e9}
                if kt_s > 0:
                    lll["kt/s"] = kt_s
                #self.logger.experiment.log(lll, step=int(real_step)) # might need to fix the real step bs
                trainer.logger.log_metrics(lll, int(real_step)) # might need to fix the real step bs
            
            if trainer.global_step % args.log_freq == 0:  # logging
                trainer.my_log.write(f"{trainer.global_step} {trainer.my_epoch_loss:.6f} {math.exp(trainer.my_epoch_loss):.4f} {trainer.my_lr:.8f} {datetime.datetime.now()} {trainer.current_epoch}\n")
                trainer.my_log.flush()
    
                trainer.my_loss_sum = 0
                trainer.my_loss_count = 0

        
        if (trainer.is_global_zero) or ('deepspeed_stage_3' in args.strategy): # save pth
            if real_step % args.epoch_step_save == 0 or not hasattr(self, "firststepsave"):
                to_save_dict = pl_module.state_dict()
                my_save(
                    args, trainer,
                    to_save_dict,
                    f"{args.proj_dir}/rwkv-{trainer.current_epoch * trainer.estimated_stepping_batches + (trainer.global_step % trainer.estimated_stepping_batches)}.pth",
                    pl_module
                )
                self.firststepsave = True

        

    def on_train_epoch_start(self, trainer, pl_module):
        args = self.args
        if pl.__version__[0]=='2':
            dataset = trainer.train_dataloader.dataset
        else:
            dataset = trainer.train_dataloader.dataset.datasets
        assert "MMapDataset" in str(dataset)
        dataset.global_rank = trainer.global_rank
        dataset.real_epoch = trainer.current_epoch
        dataset.world_size = trainer.world_size
        # print(f'########## world_size {dataset.world_size} global_rank {dataset.global_rank} real_epoch {dataset.real_epoch} ##########')

    def on_train_epoch_end(self, trainer, pl_module):
        args = self.args
        to_save_dict = {}
        if (trainer.is_global_zero) or ('deepspeed_stage_3' in args.strategy):  # save pth
            if args.data_type == 'wds_img':
                raw_dict = pl_module.state_dict()
                for k in raw_dict:
                    if k.startswith('encoder.') or k.startswith('decoder.'):
                        to_save_dict[k] = raw_dict[k]
            else:
                to_save_dict = pl_module.state_dict()
            try:
                my_save(
                    args, trainer,
                    to_save_dict,
                    f"{args.proj_dir}/rwkv-{trainer.current_epoch * trainer.total_steps}-final.pth",
                    pl_module
                )
            except Exception as e:
                print('Error\n\n', e, '\n\n')


        if trainer.is_global_zero:  # logging
            trainer.my_log.write(f"{trainer.global_step} {trainer.my_epoch_loss:.6f} {math.exp(trainer.my_epoch_loss):.4f} {trainer.my_lr:.8f} {datetime.datetime.now()} {trainer.current_epoch}\n")
            trainer.my_log.flush()

            trainer.my_loss_sum = 0
            trainer.my_loss_count = 0
            trainer.my_log.write(f"EPOCH END{args.my_timestamp}\n\n")
    
    def set_current_epoch(self, epoch: int):
        print(f"Setting current epoch to {epoch}")
        self.trainer.fit_loop.epoch_progress.current.completed = epoch
        self.trainer.fit_loop.epoch_progress.current.processed = epoch
        assert self.trainer.current_epoch == epoch, f"{self.trainer.current_epoch} != {epoch}"
    
    def set_global_step(self, global_step: int):
        print(f"Setting global step to {global_step}")
        self.trainer.fit_loop.epoch_loop.manual_optimization.optim_step_progress.total.completed = (
            global_step
        )
        self.trainer.fit_loop.epoch_loop.automatic_optimization.optim_progress.optimizer.step.total.completed = (
            global_step
        )
        
        assert self.trainer.global_step == global_step, f"{self.trainer.global_step} != {global_step}"
    
    def set_total_batch_idx(self, total_batch_idx: int):
        print(f"Setting total batch idx to {total_batch_idx}")
        self.trainer.fit_loop.epoch_loop.batch_progress.total.ready = (
            total_batch_idx + 1
        )
        self.trainer.fit_loop.epoch_loop.batch_progress.total.completed = (
            total_batch_idx
        )
        assert (
            self.total_batch_idx == total_batch_idx + 1
        ), f"{self.total_batch_idx} != {total_batch_idx + 1}"
    
    @property
    def total_batch_idx(self) -> int:
        return self.trainer.fit_loop.epoch_loop.total_batch_idx + 1



@rank_zero_only
def generate_init_weight(model, init_weight_name):
    mm = model.generate_init_weight()

    if model.args.my_pile_stage == 1:
        if len(model.args.load_model) > 0:
            print(f"Combine weights from {model.args.load_model}...")
            load_dict = torch.load(model.args.load_model, map_location="cpu")
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

    if model.args.my_pile_stage == 1:
        print("Done. Now go for stage 2.")
        exit(0)
