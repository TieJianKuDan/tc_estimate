import os
import pickle
from multiprocessing import Process

import numpy as np
import pandas as pd
import xarray as xr


def legitimate_record(record):
    return hasattr(record, "IRWIN") \
        and hasattr(record, "IRWVP") \
        and np.isnan(record.IRWIN.data).sum() == 0 \
        and np.isnan(record.IRWVP.data).sum() == 0


def load_tc(tc_dir):
    tc = []
    for root, _, files in os.walk(tc_dir):  
        for filename in files:
            tc.append(
                xr.open_dataset(os.path.join(root, filename))
            )
    tc = sorted(
        tc, key=lambda tc: tc.htime.data[0]
    )
    # eliminate redundancy
    tidy_tc = []
    i = 0
    while i < len(tc):
        candi = tc[i]
        j = i + 1
        while j < len(tc):
            if pd.Timestamp(tc[j].htime.data[0]).round('60min') \
                != pd.Timestamp(candi.htime.data[0]).round('60min'):
                break
            if not legitimate_record(candi) or (tc[j].VZA.data[0] < candi.VZA.data[0] and legitimate_record(tc[j])):
                candi = tc[j]
            j += 1
        i = j
        if legitimate_record(candi):
            tidy_tc.append(candi)
        else:
            return None
    tidy_tc = [
        [
            np.concatenate((
                ele.IRWVP.data,
                ele.IRWIN.data,
            ), axis=0),
            ele.WindSpd.data[0],
        ] for ele in tidy_tc
    ]
    return tidy_tc

def load_tc_year(tc_dir, save_path):
    print(f"process {tc_dir}")
    tc_list = []
    tc_names = os.listdir(tc_dir)
    for tc_name in tc_names:
        tc_path = os.path.join(tc_dir, tc_name)
        tc = load_tc(tc_path)
        if tc != None:
            tc_list.append(tc)
    os.makedirs(save_path, exist_ok=True)
    with open(save_path + "/" + tc_dir.split("/")[-1] + ".pk", "wb") as f:
        pickle.dump(tc_list, f)
        print(f"save tc to {save_path + '/' + tc_dir.split('/')[-1] + '.pk'}")

def load_all_tc(tc_root_dir, output_dir, start=0, worker_num=10):
    years = os.listdir(tc_root_dir)
    subset_dir = [os.path.join(tc_root_dir, year) for year in years]
    workers = [None] * worker_num
    for i in range(worker_num):
        if (start + i) >= len(subset_dir):
            break
        workers[i] = Process(
            target=load_tc_year, 
            args=(
                subset_dir[start + i],
                output_dir,
            )
        )
        workers[i].start()
    for worker in workers:
        if worker != None:
            worker.join()