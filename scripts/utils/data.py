import os
import pickle
import re
import shutil
import sys
from datetime import datetime
from math import ceil
from multiprocessing import Process

import numpy as np
import pandas as pd
import psutil
import requests
import xarray as xr
from bs4 import BeautifulSoup
from scipy import interpolate


def print_mem_use():
    pid = os.getpid()  
    proc = psutil.Process(pid)  
    memory_info = proc.memory_info()  
    print(f"RSS: {memory_info.rss / (1024 ** 3)} GB")

def process_nan(matrix:np.ndarray):
    nan_positions = np.isnan(matrix)
    x, y = np.meshgrid(np.arange(matrix.shape[1]), np.arange(matrix.shape[0]))
    x = x[~nan_positions]
    y = y[~nan_positions]
    new_values = matrix[~nan_positions]
    interpolator = interpolate.griddata(
        (x, y), new_values, 
        (np.where(nan_positions)[1], np.where(nan_positions)[0]), 
        method='nearest'
    )
    matrix[nan_positions] = interpolator

class Logger(object):  # Record the log and output to the console
    def __init__(self, filename='default.log'):
        self.terminal = sys.stdout
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def download_package(url, path):  # Download package from url and save to path
    temp_path = path + "temp.tar.gz"
    unpack_path = path + re.split(r"[\.]", url.split("/")[-1])[0]
    # confirm if it has been downloaded
    if os.path.exists(unpack_path):
        print(f"{url} has been downloaded!")
        return
    if os.path.exists(temp_path):
        os.remove(temp_path)
    response = requests.get(url, stream=True)
    with open(temp_path, 'xb') as fd:
        for chunk in response.iter_content(chunk_size=1024):
            fd.write(chunk)
    if not os.path.isfile(temp_path):
        print(f"Failed to download from {url}")
        return
    try:  
        shutil.unpack_archive(temp_path, unpack_path)
    except (KeyboardInterrupt, Exception) as e:
        if os.path.exists(unpack_path):  
            shutil.rmtree(unpack_path)  
        raise
    os.remove(temp_path)
    print(f"Succeeded to download from {url} to {unpack_path}")

def download_folders(folder_urls, save_paths):
    # Record log
    sys.stdout = Logger(
        f"logs/HURSAT-B1_{os.getpid()}" + datetime.now().strftime(r"%Y_%m_%d_%H_%M") + ".log")
    for i in range(len(folder_urls)):
        print("========================>")
        print(f"Downloading folder {folder_urls[i]}")
        response = requests.get(folder_urls[i])
        if response.status_code != 200:
            print(
                f"{response.status_code}: Failed to obtain urls from {folder_urls[i]}")
            print("<========================")
            continue
        soup = BeautifulSoup(response.text, "html.parser")
        response.close()
        a_tags = soup.table.find_all("a")[5:]
        tar_urls = [folder_urls[i] + a_tag["href"] for a_tag in a_tags]
        if len(tar_urls) == 0:
            print(f"{response.status_code}: Got nothing from {folder_urls[i]}")
            print("<========================")
            continue

        if not os.path.isdir(save_paths[i]):
            os.mkdir(save_paths[i])
        for url in tar_urls:
            download_package(url, save_paths[i])
        print("<========================")

def download_HURSAT_B1(worker_num=4):
    url = "https://www.ncei.noaa.gov/data/hurricane-satellite-hursat-b1/archive/v06/"

    # Get download urls
    response = requests.get(url)
    if response.status_code != 200:
        print(f"{response.status_code}: Failed to obtain {url}")
        exit(-1)
    soup = BeautifulSoup(response.text, "html.parser")
    response.close()
    a_tags = soup.table.find_all("a")[5:]
    folder_urls = [url + a_tag["href"] for a_tag in a_tags][-7:-1]
    save_paths = ["data/HURSAT-B1/" + a_tag["href"] for a_tag in a_tags][-7:-1]

    workers = [None] * worker_num
    task_num = ceil(len(folder_urls) / worker_num)
    for i in range(worker_num):
        start = i * task_num
        stop = min((i + 1)*task_num, len(folder_urls))
        workers[i] = Process(
            target=download_folders, 
            args=(
                folder_urls[start:stop],
                save_paths[start:stop]
            )
        )
        workers[i].start()
    for worker in workers:
        worker.join()
