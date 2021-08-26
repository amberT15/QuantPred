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


def evaluate_per_cell_line(run_path, testset, targets, log_all, choose_cell=-1):
# for each experiment
# for i, run_path in enumerate(summary_df['run_path'].values):
    # get all true and pred values
    metrics_columns = ['mse', 'scaled mse', 'JS', 'poisson NLL', 'pearson r', 'targets']
    all_truth, all_pred = get_true_pred(run_path, testset, log_all)
    if choose_cell>-1:
        assert all_truth.shape[-1] == 1, 'Wrong ground truth!'
        print(targets)
        all_pred = all_pred[:,:,choose_cell:(choose_cell+1)]

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

def evaluate_idr(run_path, log_all):
    cl_datasets = glob.glob('/home/shush/profile/QuantPred/datasets/cell_line_specific_test_sets/cell_line_*/complete/peak_centered/i_2048_w_1/')
    one_run_all = []
    for data_dir in cl_datasets:
        sts = util.load_stats(data_dir)
        testset = util.make_dataset(data_dir, 'test', sts, batch_size=512, shuffle=False)
        targets = pd.read_csv(data_dir+'targets.txt', sep='\t')['identifier'].values
        i = int(data_dir.split('cell_line_')[-1].split('/complete')[0])
        performance = evaluate_per_cell_line(run_path, testset, targets, log_all, choose_cell=i)
        one_run_all.append(performance)
    return pd.concat(one_run_all)

def summarize_project(project_name, factors, output_path, testset, targets,
                      wandb_dir='/mnt/31dac31c-c4e2-4704-97bd-0788af37c5eb/shush/wandb/*/*',
                      log_all=False, idr=False, run_list=[]):
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
            line.append(config[factor]['value'])
        print(line)
        if idr:
            one_model_df = evaluate_idr(run_dir, log_all)
        else:
            one_model_df =  evaluate_per_cell_line(run_dir, testset, targets, log_all)
        one_model_df[['run_path']+factors] = line
        run_summaries.append(one_model_df)
    summary_df = pd.concat(run_summaries)
    summary_df.to_csv(output_path, index=False)
    return summary_df

def get_true_pred(run_path, testset, log_all=False):
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
    else:
        return np.concatenate(all_truth), np.concatenate(all_pred)


if __name__ == '__main__':
    data_dir = '/home/shush/profile/QuantPred/datasets/chr8/complete/random_chop/i_2048_w_1/'
    sts = util.load_stats(data_dir)
    res_dir = 'summary_metrics_tables'
    exp_name = 'BPNET_AUGMENTATION'
    bpnet_runs = [
    '/mnt/906427d6-fddf-41bf-9ec6-c3d0c37e766f/amber/elzar_wandb/run-20210824_044431-448pikam',
 '/mnt/906427d6-fddf-41bf-9ec6-c3d0c37e766f/amber/elzar_wandb/run-20210824_044201-k6tl56p0',
 '/mnt/906427d6-fddf-41bf-9ec6-c3d0c37e766f/amber/elzar_wandb/run-20210824_033104-58mz2s78',
 '/mnt/906427d6-fddf-41bf-9ec6-c3d0c37e766f/amber/elzar_wandb/run-20210824_031215-qcpilvr0',
 '/mnt/906427d6-fddf-41bf-9ec6-c3d0c37e766f/amber/elzar_wandb/run-20210824_024633-64c5a0ba',
 '/mnt/906427d6-fddf-41bf-9ec6-c3d0c37e766f/amber/elzar_wandb/run-20210824_024631-78wlku5c',
 '/mnt/906427d6-fddf-41bf-9ec6-c3d0c37e766f/amber/elzar_wandb/run-20210824_024516-rq7qau8b',
 '/mnt/906427d6-fddf-41bf-9ec6-c3d0c37e766f/amber/elzar_wandb/run-20210824_024431-73os9chn',
 '/mnt/906427d6-fddf-41bf-9ec6-c3d0c37e766f/amber/elzar_wandb/run-20210824_024254-j7h15csm',
 '/mnt/906427d6-fddf-41bf-9ec6-c3d0c37e766f/amber/elzar_wandb/run-20210824_024054-364dibdw',
 '/mnt/906427d6-fddf-41bf-9ec6-c3d0c37e766f/amber/elzar_wandb/run-20210824_015607-r42gedu4',
 '/mnt/906427d6-fddf-41bf-9ec6-c3d0c37e766f/amber/elzar_wandb/run-20210823_230933-5cxo0vve',
 '/mnt/906427d6-fddf-41bf-9ec6-c3d0c37e766f/amber/elzar_wandb/run-20210823_222404-5px03at6',
 '/mnt/906427d6-fddf-41bf-9ec6-c3d0c37e766f/amber/elzar_wandb/run-20210823_222238-pdzaqdpx',
 '/mnt/906427d6-fddf-41bf-9ec6-c3d0c37e766f/amber/elzar_wandb/run-20210823_221552-y4aehhgo',
 '/mnt/906427d6-fddf-41bf-9ec6-c3d0c37e766f/amber/elzar_wandb/run-20210823_221005-jx2asv1c',
 '/mnt/906427d6-fddf-41bf-9ec6-c3d0c37e766f/amber/elzar_wandb/run-20210823_113336-6jqab2b5',
 '/mnt/906427d6-fddf-41bf-9ec6-c3d0c37e766f/amber/elzar_wandb/run-20210823_113337-0oiiwgur',
 '/mnt/906427d6-fddf-41bf-9ec6-c3d0c37e766f/amber/elzar_wandb/run-20210823_113336-0c8yu0xn',
 '/mnt/906427d6-fddf-41bf-9ec6-c3d0c37e766f/amber/elzar_wandb/run-20210823_113334-xkmjrf9k',
 '/mnt/906427d6-fddf-41bf-9ec6-c3d0c37e766f/amber/elzar_wandb/run-20210823_113335-mh3kyjdn',
 '/mnt/906427d6-fddf-41bf-9ec6-c3d0c37e766f/amber/elzar_wandb/run-20210823_113335-l8obhp67',
 '/mnt/906427d6-fddf-41bf-9ec6-c3d0c37e766f/amber/elzar_wandb/run-20210823_113332-i9itfjr5',
 '/mnt/906427d6-fddf-41bf-9ec6-c3d0c37e766f/amber/elzar_wandb/run-20210823_113323-7sbzgq06',
 '/mnt/906427d6-fddf-41bf-9ec6-c3d0c37e766f/amber/elzar_wandb/run-20210823_113230-p9yv3sjb',
 '/mnt/906427d6-fddf-41bf-9ec6-c3d0c37e766f/amber/elzar_wandb/run-20210823_113230-ksao2t0q',
 '/mnt/906427d6-fddf-41bf-9ec6-c3d0c37e766f/amber/elzar_wandb/run-20210823_113229-cvwwhn2l',
 '/mnt/906427d6-fddf-41bf-9ec6-c3d0c37e766f/amber/elzar_wandb/run-20210823_113228-bbde8ryd',
 '/mnt/906427d6-fddf-41bf-9ec6-c3d0c37e766f/amber/elzar_wandb/run-20210823_113228-0esuv11b',
 '/mnt/906427d6-fddf-41bf-9ec6-c3d0c37e766f/amber/elzar_wandb/run-20210823_113228-pyk4sc73',
 '/mnt/906427d6-fddf-41bf-9ec6-c3d0c37e766f/amber/elzar_wandb/run-20210823_113227-mdzlpaqy',
 '/mnt/906427d6-fddf-41bf-9ec6-c3d0c37e766f/amber/elzar_wandb/run-20210823_113226-v7s8qakb',
 '/mnt/906427d6-fddf-41bf-9ec6-c3d0c37e766f/amber/elzar_wandb/run-20210823_113225-u97f2bt5',
 '/mnt/906427d6-fddf-41bf-9ec6-c3d0c37e766f/amber/elzar_wandb/run-20210823_113224-pd85fbam',
 '/mnt/906427d6-fddf-41bf-9ec6-c3d0c37e766f/amber/elzar_wandb/run-20210823_113224-mj7maqix',
 '/mnt/906427d6-fddf-41bf-9ec6-c3d0c37e766f/amber/elzar_wandb/run-20210823_113223-9k90siyf'
    ]
    factors = ['data_dir', 'crop', 'rev_comp', 'smooth']
    csv_file_suffix = exp_name + '.csv'

    idr_result_path = os.path.join(res_dir, 'IDR_'+csv_file_suffix)
    result_path = os.path.join(res_dir, 'WHOLE_'+csv_file_suffix)

    log_all = False
    testset = util.make_dataset(data_dir, 'test', sts, batch_size=512, shuffle=False)
    targets = pd.read_csv(data_dir+'targets.txt', sep='\t')['identifier'].values
    summarize_project('toneyan/'+exp_name, factors,
                      result_path, testset, targets, log_all=log_all,
                      run_list=bpnet_runs)
    summarize_project('toneyan/'+exp_name, factors,
                      idr_result_path, None, None, log_all=log_all, idr=True,
                      run_list=bpnet_runs)
