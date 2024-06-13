import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset

from .ds import *


class PLDM(pl.LightningDataModule):
    '''
    abstract class
    '''
    def __init__(self):
        super(PLDM, self).__init__()
    
    @property
    def train_sample_num(self):
        return len(self.train_set)
    
    @property
    def val_sample_num(self):
        return len(self.val_set)
    
    @property
    def test_sample_num(self):
        return len(self.test_set)

    def train_dataloader(self):
        dl = None
        if self.num_workers == 1:
            dl = DataLoader(
                self.train_set, 
                batch_size=self.batch_size, 
                shuffle=True,
                drop_last=True
            )
        else:
            dl = DataLoader(
                self.train_set, 
                batch_size=self.batch_size, 
                shuffle=True,
                drop_last=True,
                num_workers=self.num_workers,
                persistent_workers=self.persistent_workers,
            )
        return dl

    def val_dataloader(self):
        dl = None
        if self.num_workers == 1:
            dl = DataLoader(
                self.val_set, 
                batch_size=self.batch_size, 
                shuffle=False,
                drop_last=True
            )
        else:
            dl = DataLoader(
                self.val_set, 
                batch_size=self.batch_size, 
                shuffle=False,
                drop_last=True,
                num_workers=self.num_workers,
                persistent_workers=self.persistent_workers
            )
        return dl

    def test_dataloader(self):
        dl = None
        if self.num_workers == 1:
            dl = DataLoader(
                self.test_set,
                batch_size=self.batch_size, 
                shuffle=False,
                drop_last=True
            )
        else:
            dl = DataLoader(
                self.test_set,
                batch_size=self.batch_size, 
                shuffle=False,
                drop_last=True,
                num_workers=self.num_workers,
                persistent_workers=self.persistent_workers
            )
        return dl

class HURSATB1PL(PLDM):

    def __init__(self, data_config, batch_size, seed):
        super(HURSATB1PL, self).__init__()
        self.data_dir = data_config.data_dir
        self.seq_len = data_config.seq_len
        self.interval = data_config.interval
        self.num_workers = data_config.num_workers
        # self.val_ratio = data_config.val_ratio
        self.persistent_workers = data_config.persistent_workers
        self.h, self.w = (data_config.h, data_config.w)
        self.batch_size = batch_size
        self.time_info = data_config.time_info
        self.seed = seed
        self.train_set = None
        self.val_set = None
        self.test_set = None

    def setup(self, stage:str):
        datasets = None
        if stage == "fit" \
            and (
                self.train_set == None \
                or self.val_set == None
            ):
            print("prepare train_set and val_set")
            self.train_set = HURSATB1(
                self.data_dir, 
                self.seq_len, 
                self.interval,
                self.h, self.w,
                self.time_info,
                2000, 2013, 
                balance=False, 
                seed=self.seed
            )
            self.val_set = HURSATB1(
                self.data_dir, 
                self.seq_len, 
                self.interval, 
                self.h, self.w,
                self.time_info,
                2013, 2015,
                balance=False, 
                seed=self.seed
            )
        elif stage == "test" \
            and self.test_set == None:
            print("prepare test_set")
            datasets = HURSATB1(
                self.data_dir, 
                self.seq_len, 
                self.interval, 
                self.h, self.w,
                self.time_info,
                2015, 2016, 
                balance=False, 
                seed=self.seed
            )
            self.test_set = datasets
        else:
            print(f"stage should be fit or test and current stage is {stage}")
