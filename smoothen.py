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
from test_to_bw_fast import open_bw, merge_bed, smoothen_bw, remove_nans
import repl_analysis
import time
from tqdm import tqdm
from scipy import stats
import sys, os
from tqdm import tqdm
import util
import multiprocessing as mp

def scipy_get_pr(bw_paths, bedfile='/home/shush/profile/QuantPred/bin_exp/truth/merged_truth_1_raw.bed'):

    all_vals_dict_nans = {}
    all_bws_dict = {}
    mean_vals_dict = {}
    for bw_path in bw_paths:
        all_bws_dict[bw_path] = []
        vals = repl_analysis.get_mean_per_range(bw_path, bedfile, keep_all=True)
        all_vals_dict_nans[bw_path] = np.array([v  for v_sub in vals for v in v_sub])
        all_vals_dict_1d = remove_nans(all_vals_dict_nans)
    assert len(all_vals_dict_1d[bw_paths[0]])>1 and len(all_vals_dict_1d[bw_paths[1]])>1, bw_paths
    pr = stats.pearsonr(all_vals_dict_1d[bw_paths[0]], all_vals_dict_1d[bw_paths[1]])[0]

    assert ~np.isnan(pr)
    return pr

def set_up_dirs():
    base_dir = util.make_dir('3smooth_exp')
    out_dir = util.make_dir(os.path.join(base_dir, 'pr_csv'))
    true_binned_dir = '/home/shush/profile/QuantPred/bin_exp/truth/'


    pred_binned_dir = '/home/shush/profile/QuantPred/bin_exp/pred/'
    in_bed_filename = '/home/shush/profile/QuantPred/bin_exp/truth/truth_1_raw.bed'

    truth_dir = util.make_dir(base_dir + '/truth')
    pred_dir = util.make_dir(base_dir + '/pred')
    for d in os.listdir(true_binned_dir):
        d_path = os.path.join(true_binned_dir, d)
        if os.path.isdir(d_path):
            [util.make_dir(os.path.join(folder, d)) for folder in [truth_dir, pred_dir]]


def smoothen_one_pair(bin_size):
    sigma_values = [0, 1, 5, 10, 20, 50, 100, 200, 500, 1000]
    # sigma_values = [5]
    base_dir = '3smooth_exp'
    out_dir = os.path.join(base_dir, 'pr_csv')
    true_binned_dir = '/home/shush/profile/QuantPred/bin_exp/truth/'


    pred_binned_dir = '/home/shush/profile/QuantPred/bin_exp/pred/'
    in_bed_filename = '/home/shush/profile/QuantPred/bin_exp/truth/truth_1_raw.bed'

    truth_dir = base_dir + '/truth'
    pred_dir = base_dir + '/pred'
    # for d in os.listdir(true_binned_dir):
    #     d_path = os.path.join(true_binned_dir, d)
    #     if os.path.isdir(d_path):
    #         [util.make_dir(os.path.join(folder, d)) for folder in [truth_dir, pred_dir]]


    pr_dict = {'truth_path':[], 'pred_path':[], 'cell_line_id':[], 'sigma':[], 'Pearson_R':[], 'bin_size':[]}
    for sigma_value in sigma_values:
    # bin_size, sigma_value = bin_sigma
        for file in os.listdir(true_binned_dir):
        # for file in ['1_GM23338']:

            subdirs = os.path.join(true_binned_dir, file)
            if os.path.isdir(subdirs):

                true_raw_bw_list = [os.path.join(subdirs, bw_file) for bw_file in os.listdir(subdirs) if '_{}_raw.bw'.format(bin_size) in bw_file]
                assert len(true_raw_bw_list) == 1, 'Too many or not enough bws found in {}!'.format(subdirs)
                true_raw_bw = true_raw_bw_list[0]
                pred_raw_bw = true_raw_bw.replace('truth', 'pred').replace('_{}_'.format(bin_size), '_{}_{}_'.format(bin_size, bin_size))
                cell_line_id = true_raw_bw.split('/')[-2]
                print('************************')
                print('Cell line {}, bin_size {}, sigma {} filenames {}, {}'.format(cell_line_id, bin_size, sigma_value, true_raw_bw, pred_raw_bw))

                out_dirs = [os.path.join(folder, cell_line_id) for folder in [truth_dir, pred_dir]]

                [sm_true, sm_pred] = smoothen_bw([true_raw_bw, pred_raw_bw], in_bed_filename, sigma_value, out_dirs)
                pr = scipy_get_pr([sm_true, sm_pred])

                pr_dict['truth_path'].append(sm_true)
                pr_dict['pred_path'].append(sm_pred)
                pr_dict['cell_line_id'].append(cell_line_id)
                pr_dict['sigma'].append(sigma_value)
                pr_dict['Pearson_R'].append(pr)
                pr_dict['bin_size'].append(bin_size)
    pr_df = pd.DataFrame(pr_dict)
    pr_df.to_csv(os.path.join(out_dir, '{}_{}.csv'.format(bin_size, sigma_value)))

def main():
    set_up_dirs()
    bin_sizes = [1, 32, 128, 2048]
    for bin_size in bin_sizes:
        smoothen_one_pair(bin_size)
    # with mp.Pool(processes=8) as pool:
    #     list(tqdm(pool.imap(smoothen_one_pair, bin_sizes)))

if __name__ == '__main__':
  main()


# def get_pr(bw_paths, bedfile='/home/shush/profile/QuantPred/bin_exp/truth/merged_truth_1_raw.bed'):
#     all_vals_dict_nans = {}
#     all_bws_dict = {}
#     mean_vals_dict = {}
#     for bw_path in bw_paths:
#         all_bws_dict[bw_path] = []
#         vals = repl_analysis.get_mean_per_range(bw_path, bedfile, keep_all=True)
#         all_vals_dict_nans[bw_path] = np.array([v  for v_sub in vals for v in v_sub])
#         all_vals_dict_1d = remove_nans(all_vals_dict_nans)
#         all_vals_dict = {k:np.expand_dims(np.expand_dims(v, -1), -1) for k, v in all_vals_dict_1d.items()}
#
#
#     pr = metrics.PearsonR(1)
#     pr.update_state(all_vals_dict[bw_paths[0]], all_vals_dict[bw_paths[1]])
#     pr_value = pr.result().numpy()
#     assert ~np.isnan(pr_value)
#     return pr_value
#
#
#
# true_binned_dir = '/home/shush/profile/QuantPred/bin_exp/truth/'
# pred_binned_dir = '/home/shush/profile/QuantPred/bin_exp/pred/'
# in_bed_filename = '/home/shush/profile/QuantPred/bin_exp/truth/truth_1_raw.bed'
# util.make_dir('smooth_exp')
# truth_dir = 'smooth_exp/truth'
# pred_dir = 'smooth_exp/pred'
# pr_dict = {'truth_path':[], 'pred_path':[], 'cell_line_id':[], 'sigma':[], 'Pearson_R':[], 'bin_size':[]}
#
# bin_sizes = [1, 32, 64, 128, 256, 512, 1024, 2048]
# bin_sizes = sys.argv[1]
# bin_sizes = [int(b) for b in bin_sizes.split(',')]
# print(bin_sizes)
# for bin_size in tqdm(bin_sizes):
#     for file in os.listdir(true_binned_dir):
#         subdirs = os.path.join(true_binned_dir, file)
#         if os.path.isdir(subdirs):
#
#             true_raw_bw_list = [os.path.join(subdirs, bw_file) for bw_file in os.listdir(subdirs) if '_{}_raw.bw'.format(bin_size) in bw_file]
#             assert len(true_raw_bw_list) == 1, 'Too many or not enough bws found!'
#             true_raw_bw = true_raw_bw_list[0]
#             pred_raw_bw = true_raw_bw.replace('truth', 'pred').replace('_{}_'.format(bin_size), '_{}_{}_'.format(bin_size, bin_size))
#             cell_line_id = true_raw_bw.split('/')[-2]
#             out_dirs = [util.make_dir(os.path.join(folder, cell_line_id)) for folder in [truth_dir, pred_dir]]
#     #         if len(os.listdir(out_dirs[0]))==0:
#     #             print(out_dirs[0])
#             for sigma_value in tqdm([0, 1, 5, 10, 20, 50, 100, 200, 500, 1000]):
#                 [sm_true, sm_pred] = smoothen_bw([true_raw_bw, pred_raw_bw], in_bed_filename, sigma_value, out_dirs)
#                 pr = get_pr([sm_true, sm_pred])
#
#                 pr_dict['truth_path'].append(sm_true)
#                 pr_dict['pred_path'].append(sm_pred)
#                 pr_dict['cell_line_id'].append(cell_line_id)
#                 pr_dict['sigma'].append(sigma_value)
#                 pr_dict['Pearson_R'].append(pr)
#                 pr_dict['bin_size'].append(bin_size)
#
#
# pr_df = pd.DataFrame(pr_dict)
# # print(pr_df)
# pr_df.to_csv('smooth_exp/summary_all_models.csv')
