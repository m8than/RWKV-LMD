########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import json, math, random, os, sys
import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch_lightning.utilities import rank_zero_info
from .binidx import MMapIndexedDataset
from .utils import MaybeIsPrime

# Temporary fix dataset resuming by randomising dataset
class MMapDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.data = MMapIndexedDataset(args.data_file)
        self.data_size = len(self.data._bin_buffer) // self.data._index._dtype_size
        self.count = self.data_size // args.ctx_len
        self.ctx_len = args.ctx_len
        self.args.dataset_len = self.count

        # for random sampling
        global_rank = self.args.trainer.global_rank
        current_time = int(time.time() * 1000)
        seed_value = global_rank + current_time
        np.random.seed(seed_value)

    def __len__(self):
        return self.count
         
    def __getitem__(self, idx):
        if self.args.random_data:
            i = np.random.randint(0, self.data_size - (self.ctx_len+1))
            data_chunk = self.data.get(idx=0, offset=i, length=self.ctx_len + 1).astype(int)
        else:
            data_chunk = self.data.get(idx=0, offset=idx * self.ctx_len, length=self.ctx_len + 1).astype(int)
            
        x = torch.tensor(data_chunk[:-1], dtype=torch.long)
        y = torch.tensor(data_chunk[1:], dtype=torch.long)
        return x, y
