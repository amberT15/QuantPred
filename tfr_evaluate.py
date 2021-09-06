#!/usr/bin/env python
import tensorflow as tf
import util
from test_to_bw_fast import read_model
import metrics
import wandb
from test_to_bw_fast import get_config
import glob, os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from test_to_bw_fast import open_bw
import numpy as np
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from tqdm import tqdm
from scipy.spatial import distance
from scipy import stats
import pickle
from metrics import np_poiss, np_mse, get_scaled_mse, get_js_dist, np_pr


def evaluate_per_cell_line(run_path, testset, targets, log_all, log_truth, choose_cell=-1):
# for each experiment
# for i, run_path in enumerate(summary_df['run_path'].values):
    # get all true and pred values
    metrics_columns = ['mse', 'scaled mse', 'JS', 'poisson NLL', 'pearson r', 'targets']
    all_truth, all_pred = get_true_pred(run_path, testset, log_all, log_truth)

    if choose_cell>-1:
        # assert all_truth.shape[-1] == 1, 'Wrong ground truth!'
        # all_truth = np.expand_dims(all_truth[:,:,choose_cell])
        # print(targets[choose_cell])
        all_pred = np.expand_dims(all_pred[:,:,choose_cell], axis=-1)
    print(all_truth.shape)
    print(all_pred.shape)
    # compute per sequence mse and JS for cell line 2
    mse = np_mse(all_truth, all_pred).mean(axis=1).mean(axis=0)
    js = get_js_dist(all_truth, all_pred).mean(axis=0)
    scaled_mse = get_scaled_mse(all_truth, all_pred).mean(axis=0)
    poiss = np_poiss(all_truth, all_pred).mean(axis=1).mean(axis=0)
    pr = np_pr(all_truth, all_pred)

    performance = np.array([mse, scaled_mse, js, poiss, pr, targets])
    one_model_df = pd.DataFrame(performance.T)
    one_model_df.columns = metrics_columns
    return one_model_df

def evaluate_idr(run_path, log_all, log_truth):
    cl_datasets = glob.glob('/home/shush/profile/QuantPred/datasets/cell_line_specific_test_sets/cell_line_*/complete/peak_centered/i_2048_w_1/')
    one_run_all = []
    for data_dir in cl_datasets:
        sts = util.load_stats(data_dir)
        testset = util.make_dataset(data_dir, 'test', sts, batch_size=512, shuffle=False)
        targets = pd.read_csv(data_dir+'targets.txt', sep='\t')['identifier'].values
        i = int(data_dir.split('cell_line_')[-1].split('/complete')[0])
        performance = evaluate_per_cell_line(run_path, testset, targets, log_all, log_truth, choose_cell=i)
        one_run_all.append(performance)
    return pd.concat(one_run_all)

def summarize_project(project_name, factors, output_path, testset, targets,
                      wandb_dir='/mnt/31dac31c-c4e2-4704-97bd-0788af37c5eb/shush/wandb/*/*',
                      idr=False, run_list=[]):
    run_summaries = []
    if len(run_list)==0:
        wandb.login()
        api = wandb.Api()
        runs = api.runs(project_name)
    else:
        runs = run_list
    for run in runs:
        if len(run_list)==0:
            run_dir = glob.glob(wandb_dir+run.id)[0]
        else:
            run_dir = run
        line = [run_dir]
        config = get_config(run_dir)
        config[factors[0]]['value']
        for factor in factors:
            if factor in config.keys():
                line.append(config[factor]['value'])
            #     log_all = False
            #     log_truth = True
            #     # log_all=True
            # else:
            #     line.append('None')
            #     log_truth = False
            #     log_all = True
            #     print('Not logged RUN!')
        if ('basenjimod' in line) or ('basenjiw1' in line):
            log_all = False
            log_truth = False
            print(line)
            if idr:
                one_model_df = evaluate_idr(run_dir, log_all, log_truth)
            else:
                one_model_df =  evaluate_per_cell_line(run_dir, testset, targets, log_all, log_truth)
            one_model_df[['run_path']+factors] = line
            run_summaries.append(one_model_df)
    summary_df = pd.concat(run_summaries)
    summary_df.to_csv(output_path, index=False)
    return summary_df

def get_true_pred(run_path, testset, log_all=False, log_truth=False):
    model, bin_size = read_model(run_path, compile_model=False)
    all_truth = []
    all_pred = []
    for i, (x, y) in tqdm(enumerate(testset)):
        p = model.predict(x)
        binned_y = util.bin_resolution(y, bin_size)
        y = binned_y.numpy()
        all_truth.append(y)
        all_pred.append(p)
    if log_all:
        print('LOGGING TRUE AND PRED!!!')
        return np.log(np.concatenate(all_truth)+1), np.log(np.concatenate(all_pred)+1)
    elif log_truth:
        print('LOGGING ONLY TRUE!!!')
        return np.log(np.concatenate(all_truth)+1), np.concatenate(all_pred)
    else:
        return np.concatenate(all_truth), np.concatenate(all_pred)


if __name__ == '__main__':
    data_dir = '/home/shush/profile/QuantPred/datasets/chr8/complete/random_chop/i_2048_w_1/'
    sts = util.load_stats(data_dir)
    res_dir = 'summary_metrics_tables'
    exp_name = 'LOSS_FUNCTION'
    factors = ['model_fn', 'bin_size', 'data_dir']
    csv_file_suffix = exp_name + '.csv'

    idr_result_path = os.path.join(res_dir, 'IDR_'+csv_file_suffix)
    result_path = os.path.join(res_dir, 'WHOLE_'+csv_file_suffix)

    log_all = False
    testset = util.make_dataset(data_dir, 'test', sts, batch_size=512, shuffle=False)
    targets = pd.read_csv(data_dir+'targets.txt', sep='\t')['identifier'].values
    summarize_project('toneyan/'+exp_name, factors,
                      result_path, testset, targets)
    summarize_project('toneyan/'+exp_name, factors,
                      idr_result_path, None, None, idr=True)
