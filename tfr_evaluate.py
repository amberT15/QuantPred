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

def eval_at_resolution():
    # N, L, C  = truth_6k.shape
    # eval_bin = 2048
    # binned_truth = truth_6k.reshape(N, 2048*3//eval_bin, eval_bin//bin_size_orig, C).mean(axis=2)
    # N, L, C  = pred_6k.shape
    # binned_pred = pred_6k.reshape(N, 2048*3//eval_bin, eval_bin//bin_size_orig, C).mean(axis=2)
    pass

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

def get_performance(all_truth, all_pred, targets, testset_type):
    assert all_truth.shape[-1] == all_pred.shape[-1], 'Incorrect number of cell lines for true and pred'
    mse = metrics.get_mse(all_truth, all_pred).mean(axis=1).mean(axis=0)
    js_per_seq = metrics.get_js_per_seq(all_truth, all_pred).mean(axis=0)
    js_conc = metrics.get_js_concatenated(all_truth, all_pred)
    poiss = metrics.get_poiss_nll(all_truth, all_pred).mean(axis=1).mean(axis=0)
    pr = choose_pr_func(testset_type)(all_truth, all_pred)
    performance = {'mse': mse, 'js_per_seq': js_per_seq, 'js_conc': js_conc,
                    'poiss': poiss, 'pr': pr, 'targets':targets}
    return pd.DataFrame(performance)

def get_scaling_factors(all_truth, all_pred):
    N, L, C = all_pred.shape
    flat_pred = all_pred.reshape(N*L, C)
    flat_truth = all_truth.reshape(N*L, C)
    truth_per_cell_line_sum = flat_truth.sum(axis=0)
    pred_per_cell_line_sum = flat_pred.sum(axis=0)
    scaling_factors =  truth_per_cell_line_sum / pred_per_cell_line_sum
    return scaling_factors

def get_performance_raw_scaled(truth, targets, pred_labels, eval_type):
    complete_performance = []
    for label, pred in pred_labels.items():
        # get performance df
        performance = get_performance(truth, pred, targets, eval_type)
        performance['pred type'] = label
        performance['eval type'] = eval_type
        complete_performance.append(performance)
    return pd.concat(complete_performance)


def evaluate_run_whole(run_path, testset, targets):
    # make predictions
    truth, raw_pred = get_true_pred(run_path, testset)
    # get scales predictions
    scaling_factors = get_scaling_factors(truth, raw_pred)
    scaled_pred = raw_pred * scaling_factors
    complete_performance = get_performance_raw_scaled(truth, targets, {'raw': raw_pred,
                                                      'scaled': scaled_pred},
                                                      'whole')
    return (complete_performance, scaling_factors)

def extract_datasets(path_pattern='/mnt/1a18a49e-9a31-4dbf-accd-3fb8abbfab2d/shush/15_IDR_test_sets_6K/cell_line_*/i_6144_w_1/'):
    paths = glob.glob(path_pattern)
    assert len(paths)>0
    target_dataset = {}
    for path in paths:
        sts = util.load_stats(path)
        testset_6K = util.make_dataset(path, 'test', sts, batch_size=512, shuffle=False)
        target = pd.read_csv(path+'targets.txt', sep='\t')['identifier'].values[0]
        i = [f for f in path.split('/') if 'cell_line' in f][0].split('_')[-1]
        testset_2K = testset_6K.map(lambda x,y: (split_into_2k_chunks(x), split_into_2k_chunks(y)))
        target_dataset[(i, target)] = testset_2K
    return target_dataset

def evaluate_run_idr(run_path, target_dataset, scaling_factors):
    for (i, target), one_testset in target_dataset.items():
        # make predictions and slice the cell line
        truth, all_pred = get_true_pred(run_path, one_testset)
        raw_pred = np.expand_dims(all_pred[:,:,i], axis=-1)
        truth_6k = combine_into_6k_chunks(truth)
        raw_pred_6k = combine_into_6k_chunks(raw_pred)
        # make scaled predictions
        scaled_pred = raw_pred_6k * scaling_factors[i]
        # get idr performance raw, scaled
        assert truth_6k.shape == raw_pred_6k.shape, 'shape mismatch!'
        complete_performance = get_performance_raw_scaled(truth_6k, [target], {'raw': raw_pred_6k,
                                                          'scaled': scaled_pred},
                                                          'idr')

    return complete_performance

def get_run_metadata(run_dir):
    config = get_config(run_dir)
    relevant_config = {k:[config[k]['value']] for k in config.keys() if k not in ['wandb_version', '_wandb']}
    return pd.DataFrame(relevant_config)

def evaluate_run_whole_idr(run_dir, testset, targets, target_dataset_idr):
    # get performance for the whole chromosome
    complete_performance_whole, scaling_factors = evaluate_run_whole(run_dir, testset, targets)
    # get performance for the IDR regions only
    complete_performance_idr = evaluate_run_idr(run_dir, target_dataset_idr, scaling_factors)
    # get metadata for the run
    metadata = get_run_metadata(run_dir)
    # add metadata to performance dataframes
    combined_performance = pd.concat([complete_performance_whole, complete_performance_idr]).reset_index()
    n_rows = combined_performance.shape[0]
    metadata_broadcasted = pd.DataFrame(np.repeat(metadata.values, n_rows, axis=0))
    combined_performance_w_metadata = pd.concat([combined_performance, metadata_broadcasted], axis=1)
    return combined_performance_w_metadata



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
