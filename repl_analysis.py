#!/usr/bin/env python
# coding: utf-8


import pyBigWig, os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.gridspec as gridspec
import seaborn as sns
import shutil, wandb
import itertools
import numpy as np
# from test_to_bw import *
from test_to_bw_fast import process_run, get_vals_per_range, remove_nans
import util
import metrics
from util import SeabornFig2Grid
# N_cell_line = 12
# run_dir ='/home/shush/profile/QuantPred/wandb/run-20210702_234001-u3vwup7j/'


def summarize_replicates(run_dir, N_cell_line, bin_size):
    label_dict = {'threshold_2':{'bw':'{}_thresh2.bw'.format(bin_size), 'bed':'thresh2.bed'},
                  'raw':{'bw':'_{}_raw.bw'.format(bin_size), 'bed':'raw.bed'},
                  'IDR':{'bw': '{}_idr.bw'.format(bin_size), 'bed':'idr.bed'}
                 }

    # get the cell line specific directory
    bigwigs_dir = os.path.join(run_dir, 'bigwigs')
    folders_with_cell_line_N = [f for f in os.listdir(bigwigs_dir) if f.split('_')[0]==str(N_cell_line)]
    print(folders_with_cell_line_N)
    assert len(folders_with_cell_line_N) == 1, 'Folders for one cell line not identified!'
    res_dir = os.path.join(bigwigs_dir, folders_with_cell_line_N[0])
    cell_line_name = folders_with_cell_line_N[0].split('_')[1]
    plt_dir = util.make_dir(os.path.join(run_dir, 'plots_and_tables'))
    for title, fileid in label_dict.items():
        print(res_dir)
        bw_paths = [os.path.join(res_dir, f) for f in os.listdir(res_dir) if fileid['bw'] in f]
        bed_path = [os.path.join(res_dir, f) for f in os.listdir(res_dir) if fileid['bed'] in f][0]
        print('*********')
        print(len(bw_paths))
        print(bw_paths)
        assert not(len(bw_paths)<2), 'Only 1 bigwig found!'
        assert not(len(bw_paths)>4), 'Too many bigwigs found!'
        if len(bw_paths)==4:
            x_axes, y_axes = (2, 3)
            comb_of_columns = [('pred', 'truth'), ('pred', 'r2'), ('pred', 'r12'), ('truth', 'r2'), ('truth', 'r12'), ('r12', 'r2')]
        elif len(bw_paths)==3:
            x_axes, y_axes = (1, 3)
            comb_of_columns = [('pred', 'truth'), ('pred', 'r2'), ('truth', 'r2')]
        elif len(bw_paths)==2:
            x_axes, y_axes = (1, 1)
            comb_of_columns = [('pred', 'truth')]
        all_vals_dict_nans = {}
        mean_vals_dict = {}
        for bw_path in bw_paths:
            key = os.path.basename(bw_path).split('.bw')[0].split('_')[1]
            vals = get_vals_per_range(bw_path, bed_path, keep_all=True)
            all_vals_dict_nans[key] = np.array([v  for v_sub in vals for v in v_sub])
            all_vals_dict_1d = remove_nans(all_vals_dict_nans)
            # all_vals_dict = {k:np.expand_dims(np.expand_dims(v for k, v in all_vals_dict_1d.items()}
            mean_vals_dict[key] = np.array([np.mean(v) for v in vals])
        mean_cov = pd.DataFrame(mean_vals_dict)
        joint_grids = []
        titles = []
        pr_vals = {}
        for x_lab, y_lab in comb_of_columns:
            pr_result = stats.pearsonr(all_vals_dict_1d[x_lab], all_vals_dict_1d[y_lab])[0]
            if x_lab in pr_vals.keys():
                pr_vals[x_lab][y_lab] = pr_result
            else:
                pr_vals[x_lab] = {}
                pr_vals[x_lab][y_lab] = pr_result
            assert ~np.isnan(pr_vals[x_lab][y_lab]), 'NA in Pearson R!'
            titles.append('Pearson r = {}'.format(pr_vals[x_lab][y_lab]))
            if (x_lab == 'pred') or (y_lab == 'pred'):
                c = 'red'
            else:
                c = 'purple'
            joint_grid = sns.jointplot(data=mean_cov, x=x_lab, y=y_lab, color=c, kind="reg", joint_kws = {'scatter_kws':dict(alpha=0.3)})
            # Draw a line of x=y
            x0, x1 = joint_grid.ax_joint.get_xlim()
            y0, y1 = joint_grid.ax_joint.get_ylim()
            lims = [max(x0, y0), min(x1, y1)]
            joint_grid.ax_joint.plot(lims, lims, '--k')
            joint_grids.append(joint_grid)

        # A JointGrid
        fig = plt.figure(figsize=(20, 10))

        gs = gridspec.GridSpec(x_axes, y_axes)
        for i, g in enumerate(joint_grids):
            mg = SeabornFig2Grid(g, fig, gs[i])

        plt.suptitle(title+' peaks')
        gs.tight_layout(fig)
        output_prefix = '{}_{}_{}'.format(N_cell_line, cell_line_name, title)
        plt.savefig(os.path.join(plt_dir, output_prefix + '.svg'))
        df = pd.DataFrame.from_dict(pr_vals, orient='index').T
        df.to_csv(os.path.join(plt_dir, output_prefix + '.csv'))

def main():
    N_cell_lines = range(15)
    # wandb_path = '/home/shush/profile/QuantPred/wandb'
    # run_paths = []
    # wandb.login()
    # api = wandb.Api()
    # runs = api.runs('toneyan/PEAK_VS_RANDOM_BESTBASENJI')
    # for run in runs:
    #     folder_list = [os.path.join(wandb_path, folder) for folder in os.listdir(wandb_path) if run.id in folder]
    #     assert len(folder_list) == 1, 'Too many or not enough runs with same ID!'
    #     run_paths.append(folder_list[0])
    # run_paths = [run_path for run_path in run_paths if 'best_model.h5' in os.listdir(run_path+'/files')]


    # run_dir = sys.argv[1]
    # run_dir = '/home/shush/profile/QuantPred/wandb/run-20210702_234001-u3vwup7j/'
    # run_paths = ['/home/shush/profile/QuantPred/wandb/run-20210729_110628-8368urpg',
                # '/home/shush/profile/QuantPred/wandb/run-20210729_110608-rnil3owe']
    run_paths = ['run-20210728_102932-z4wk24jl']
    for run_dir in run_paths:
        # process_run(run_dir)
        for N_cell_line in N_cell_lines:
            plot_and_pr(run_dir, N_cell_line, 1)

if __name__ == '__main__':
  main()
