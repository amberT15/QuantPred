#!/usr/bin/env python

import pyBigWig, os
import numpy as np
import pandas as pd
import subprocess
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
import util, sys
import wandb
from test_to_bw_fast import get_mean_per_range, get_list_pr
import time
from tqdm import tqdm
from scipy import stats
import sys, os
from tqdm import tqdm
import util
import multiprocessing as mp
from scipy.ndimage import gaussian_filter1d

def smooth_one_binsize_one_cell_line(file, bin_size=512, out_dir='gauss_smooth'):
    print('Smoothening bin size {}'.format(bin_size))
    summary_results = [] #bin_size, sigma, pearson r, cell_line
    # grab one bin size bw (e.g. 1)
    true_binned_dir = '/home/shush/profile/QuantPred/bin_exp/truth/'
    pred_binned_dir = '/home/shush/profile/QuantPred/bin_exp/pred/'
    in_bed_filename = '/home/shush/profile/QuantPred/bin_exp/truth/merged_truth_1_raw.bed'
    subdirs = os.path.join(true_binned_dir, file)
    print('Porcessing cell line {}'.format(file))
    true_raw_bw_list = [os.path.join(subdirs, bw_file) for bw_file in os.listdir(subdirs) if '_{}_raw.bw'.format(bin_size) in bw_file]
    assert len(true_raw_bw_list) == 1, 'Too many or not enough bws found in {}!'.format(subdirs)
    true_raw_bw = true_raw_bw_list[0]
    pred_raw_bw = true_raw_bw.replace('truth', 'pred').replace('_{}_'.format(bin_size), '_{}_{}_'.format(bin_size, bin_size))
    assert os.path.isfile(pred_raw_bw), 'Pred bw {} not found!'.format(pred_raw_bw)
    # load bws for truth and pred into memory
    true_raw_np = get_mean_per_range(true_raw_bw, in_bed_filename)
    pred_raw_np = get_mean_per_range(pred_raw_bw, in_bed_filename)
    # compute pr
    pr = get_list_pr(true_raw_np, pred_raw_np)
    summary_results.append([bin_size, 0, pr, file])
    # convert to all sigmas sequentially and compute pr-s
    for sigma_value in tqdm([1, 5, 10, 20, 50, 100, 200, 500, 1000]):
        print(sigma_value)
        true_smoothened = []
        pred_smoothened = []
        for i in range(len(true_raw_np)):
            true_smoothened.append(gaussian_filter1d(true_raw_np[i], sigma_value))
            pred_smoothened.append(gaussian_filter1d(pred_raw_np[i], sigma_value))
        smooth_pr = get_list_pr(true_smoothened, pred_smoothened)
        summary_results.append([bin_size, sigma_value, smooth_pr, file])
    df = pd.DataFrame(summary_results, columns=['bin_size', 'sigma', 'pearson_r', 'cell_line'])
    df.to_csv(os.path.join(out_dir, 'bin_size_{}_{}.csv'.format(bin_size, file)))



def main():
    out_dir = util.make_dir('gauss_smooth')

    true_binned_dir = '/home/shush/profile/QuantPred/bin_exp/truth/'
    pred_binned_dir = '/home/shush/profile/QuantPred/bin_exp/pred/'
    in_bed_filename = '/home/shush/profile/QuantPred/bin_exp/truth/merged_truth_1_raw.bed'
    files = []
    for file in tqdm(os.listdir(true_binned_dir)):
        subdirs = os.path.join(true_binned_dir, file)
        if os.path.isdir(subdirs):
            files.append(file)
    with mp.Pool(processes=15) as pool:
        list(tqdm(pool.imap(smooth_one_binsize_one_cell_line, files)))

if __name__ == '__main__':
  main()
