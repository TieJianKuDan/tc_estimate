import os
import pickle
import random

import numpy as np
from scipy.ndimage import zoom 
from torch.utils.data import Dataset


def crop(x, h, w):
    zoom_factor = (1, 1, h/301, w/301)
    scaled_matrix = zoom(x, zoom_factor, order=1)
    start_row = (scaled_matrix.shape[0] - h) // 2  
    start_col = (scaled_matrix.shape[1] - w) // 2
    return scaled_matrix[start_row:start_row+h, start_col:start_col+w]

class HURSATB1(Dataset):

    def __init__(
            self, 
            root_dir, 
            seq_len, 
            interval, 
            h, w,
            time_info=False,
            start=1978, 
            end=2015, 
            balance=True,
            seed=2333
        ):
        '''
        end year is not included
        '''
        super(HURSATB1, self).__init__()
        self.tcs = []
        self.seq_len = seq_len
        self.interval = interval
        self.start = start
        self.end = end
        self.h, self.w = (h, w)
        self.time_info = time_info
        for year in range(start, end):
            with open(os.path.join(root_dir, f"{year}.pk"), "rb") as f:
                self.tcs += pickle.load(f)
        self.balance = balance
        if balance:
            self.tc_seqs = self.balance_seq(seed)
        else:
            self.tc_seqs = self.no_balance_seq(time_info)

    def __len__(self) -> int:
        return len(self.tc_seqs)

    def calc_pd(self):
        os.makedirs(".cache", exist_ok=True)
        cache_path = f".cache/pd_{self.start}_{self.end}_{self.seq_len}_{self.interval}.pk"
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
            return cache
        wss = []
        for tc in self.tcs:
            if len(tc) <= (self.seq_len - 1) * self.interval:
                continue
            for i in range((self.seq_len - 1) * self.interval, len(tc)):
                rad = [
                    ele[0] for ele \
                        in tc[i - (self.seq_len - 1) * self.interval:i+1:self.interval]
                ]
                nan = sum([np.isnan(ele).sum() for ele in rad]) > 0
                if nan:
                    continue
                ws = tc[i][1]
                if ws < 0 or ws >= 137:
                    continue
                wss.append(ws)
        pd = np.zeros((137))
        for ws in wss:
            pd[round(ws)] += 1
        mean = round(sum(wss) / len(wss))
        with open(cache_path, "wb") as f:
            pickle.dump((pd, mean), f)
        return pd, mean

    def balance_seq(self, seed):
        random.seed(seed)
        pd, mean = self.calc_pd()
        mean = 60
        tc_seqs = []
        for tc in self.tcs:
            if len(tc) <= (self.seq_len - 1) * self.interval:
                continue
            for i in range((self.seq_len - 1) * self.interval, len(tc)):
                rad = [
                    ele[0] for ele \
                        in tc[i - (self.seq_len - 1) * self.interval:i+1:self.interval]
                ]
                nan = sum([np.isnan(ele).sum() for ele in rad]) > 0
                if nan:
                    continue
                ws = tc[i][1]
                if ws < 0 or ws >= 137:
                    continue
                if pd[round(ws)] >= pd[mean]:
                    if random.random() < (pd[mean] / pd[round(ws)]):
                        tc_seqs.append([rad, ws, 0])
                else:
                    for j in range(round(pd[mean] / pd[round(ws)])):
                        tc_seqs.append([rad, ws, j % 4])
                    if random.random() < (pd[mean] % pd[round(ws)]) / pd[round(ws)]:
                        tc_seqs.append([rad, ws, (j + 1) % 4])
        return tc_seqs
    
    def no_balance_seq(self, time_info):
        tc_seqs = []
        for tc in self.tcs:
            if len(tc) <= (self.seq_len - 1) * self.interval:
                continue
            for i in range((self.seq_len - 1) * self.interval, len(tc)):
                rad = [
                    ele[0] for ele \
                        in tc[i - (self.seq_len - 1) * self.interval:i+1:self.interval]
                ]
                nan = sum([np.isnan(ele).sum() for ele in rad]) > 0
                if nan:
                    continue
                ws = tc[i][1]
                if ws < 0 or ws >= 137:
                    continue
                if time_info:
                    tc_seqs.append([rad, i*3.0, ws, 0])
                else:
                    tc_seqs.append([rad, ws, 0])
        return tc_seqs

    def __getitem__(self, index: int):
        # (t, c, h, w)
        rad = np.stack(self.tc_seqs[index][0], axis=0)
        rad = rad[:, :, 22 : 278, 22 : 278]
        # rad = crop(rad, self.h, self.w)
        if self.balance:
            rad = np.rot90(rad, k=self.tc_seqs[index][2], axes=[2, 3]).copy()
        if self.time_info:
            return (
                rad,
                self.tc_seqs[index][1], # time info
                self.tc_seqs[index][2]
            )
        else:
            return (
                rad,
                self.tc_seqs[index][1],
            )
