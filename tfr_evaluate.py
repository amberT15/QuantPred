#!/usr/bin/env python
import tensorflow as tf
import wandb
import glob, os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy.spatial import distance
from scipy import stats
from test_to_bw_fast import get_config, read_model
import util
import metrics

def get_true_pred(run_path, testset):
    model, bin_size = read_model(run_path, compile_model=False)
    all_truth = []
    all_pred = []
    for i, (x, y) in tqdm(enumerate(testset)):
        p = model.predict(x)
        binned_y = util.bin_resolution(y, bin_size)
        y = binned_y.numpy()
        all_truth.append(y)
        all_pred.append(p)
    return np.concatenate(all_truth), np.concatenate(all_pred)

def split_into_2k_chunks(x, input_size=2048):
    N = tf.shape(x)[0]
    L = tf.shape(x)[1]
    C = tf.shape(x)[2]
    x_4D = tf.reshape(x, (N, L//input_size, input_size, C))
    x_split_to_2k = tf.reshape(x_4D, (N*L//input_size, input_size, C))
    return x_split_to_2k

def combine_into_6k_chunks(x, chunk_number=3):
    N, L, C = x.shape
    x_6k = np.reshape(x, (N//chunk_number, chunk_number*L, C))
    return x_6k

def choose_pr_func(testset_type):
    if testset_type == 'whole':
        get_pr = metrics.get_pearsonr_concatenated
    elif testset_type == 'idr':
        get_pr = metrics.get_pearsonr_per_seq
    return get_pr

def get_performance(all_truth, all_pred, testset_type):
    assert all_truth.shape[-1] == all_pred.shape[-1], 'Incorrect number of cell lines for true and pred'
    mse = metrics.get_mse(all_truth, all_pred).mean(axis=1).mean(axis=0)
    js_per_seq = metrics.get_js_per_seq(all_truth, all_pred).mean(axis=0)
    js_conc = metrics.get_js_concatenated(all_truth, all_pred)
    poiss = metrics.get_poiss_nll(all_truth, all_pred).mean(axis=1).mean(axis=0)
    pr = choose_pr_func(testset_type)(all_truth, all_pred)
    performance = {'mse': mse, 'js_per_seq': js_per_seq, 'js_conc': js_conc,
                    'poiss': poiss, 'pr': pr}
    return pd.DataFrame(performance)


def evaluate_run_whole(run_path, testset):
    # make predictions
    # scale predictions
    # for both raw and scaled predictions and truth:
    # get performance df
    # add if raw or scaled
    # add run info
    # return complete run df
    pass

def evaluate_run_idr(run_path, testsets_15, scaling_factors):
    # for each testset
    # make predictions and slice the cell line
    # make scaled predictions
    # get idr performance raw, scaled
    # add column for if raw or scaled
    # add run info
    # return complete run df
    pass

def evaluate_idr(run_path,
                 cell_line_6k_dataset_paths='/home/shush/profile/QuantPred/datasets/15_IDR_test_sets_6K/cell_line_*/i_6144_w_1/'):
    cl_datasets = glob.glob(cell_line_6k_dataset_paths)
    one_run_all = []
    for data_dir in cl_datasets:
        sts = util.load_stats(data_dir)
        testset = util.make_dataset(data_dir, 'test', sts, batch_size=512, shuffle=False)
        targets = pd.read_csv(data_dir+'targets.txt', sep='\t')['identifier'].values
        i = int(data_dir.split('cell_line_')[-1].split('/complete')[0])
        performance = evaluate_per_cell_line(run_path, testset, targets, choose_cell=i)
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
        print(line)
        if idr:
            one_model_df = evaluate_idr(run_dir)
        else:
            one_model_df =  evaluate_per_cell_line(run_dir, testset, targets)
        one_model_df[['run_path']+factors] = line
        run_summaries.append(one_model_df)
    summary_df = pd.concat(run_summaries)
    summary_df.to_csv(output_path, index=False)
    return summary_df




if __name__ == '__main__':
    data_dir = '/home/shush/profile/QuantPred/datasets/chr8/complete/random_chop/i_2048_w_1/'
    sts = util.load_stats(data_dir)
    res_dir = 'summary_metrics_tables'
    exp_name = 'AUGMENTATION_BIN_SIZE'
    factors = ['model_fn', 'bin_size', 'data_dir', 'crop', 'rev_comp']
    csv_file_suffix = exp_name + '.csv'

    idr_result_path = os.path.join(res_dir, 'IDR_'+csv_file_suffix)
    result_path = os.path.join(res_dir, 'WHOLE_'+csv_file_suffix)

    testset = util.make_dataset(data_dir, 'test', sts, batch_size=512, shuffle=False)
    targets = pd.read_csv(data_dir+'targets.txt', sep='\t')['identifier'].values
    # summarize_project('toneyan/'+exp_name, factors,
    #                   result_path, testset, targets)
    summarize_project('toneyan/'+exp_name, factors,
                      idr_result_path, None, None, idr=True)
