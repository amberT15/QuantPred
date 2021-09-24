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

def change_resolution(truth, bin_size_orig, eval_bin):
    N, L, C  = truth.shape
    binned_truth = truth.reshape(N, L*bin_size_orig//eval_bin, eval_bin//bin_size_orig, C).mean(axis=2)
    return (binned_truth)

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
    try:
        pr = choose_pr_func(testset_type)(all_truth, all_pred)
    except ValueError:
        pr = [np.nan for i in range(len(poiss))]
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
    if (np.isfinite(scaling_factors)).sum() == len(scaling_factors): # if all factors are ok
        scaled_pred = raw_pred * scaling_factors
        sets_to_process = {'raw': raw_pred, 'scaled': scaled_pred}
    else:
        sets_to_process = {'raw': raw_pred}
    complete_performance = get_performance_raw_scaled(truth, targets,
                                                      sets_to_process, 'whole')
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
        target_dataset[(int(i), target)] = testset_2K
    return target_dataset

def evaluate_run_idr(run_path, target_dataset, scaling_factors):
    complete_performance = []
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
        complete_performance.append(get_performance_raw_scaled(truth_6k, [target], {'raw': raw_pred_6k,
                                                          'scaled': scaled_pred},
                                                          'idr'))

    return pd.concat(complete_performance)

def get_run_metadata(run_dir):
    config = get_config(run_dir)
    relevant_config = {k:[config[k]['value']] for k in config.keys() if k not in ['wandb_version', '_wandb']}
    return pd.DataFrame(relevant_config)

def collect_datasets(data_dir='/home/shush/profile/QuantPred/datasets/chr8/complete/random_chop/i_2048_w_1/'):
    # get testset
    sts = util.load_stats(data_dir)
    testset = util.make_dataset(data_dir, 'test', sts, batch_size=512, shuffle=False)
    targets = pd.read_csv(data_dir+'targets.txt', sep='\t')['identifier'].values
    # get cell line specific IDR testsets in 6K
    target_dataset_idr = extract_datasets()
    return (testset, targets, target_dataset_idr)

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
    metadata_broadcasted = pd.DataFrame(np.repeat(metadata.values, n_rows, axis=0), columns=metadata.columns)
    combined_performance_w_metadata = pd.concat([combined_performance, metadata_broadcasted], axis=1)
    # save scaling factors
    scaling_factors_per_cell = pd.DataFrame(zip(targets, scaling_factors,
                                            [run_dir for i in range(len(scaling_factors))]))
    return (combined_performance_w_metadata, scaling_factors_per_cell)

def process_run_list(run_dirs, output_summary_filepath):
    # get datasets
    testset, targets, target_dataset_idr = collect_datasets()
    # process runs
    all_run_summaries = []
    all_scale_summaries = []
    for run_dir in run_dirs:
        print(run_dir)
        run_summary, scale_summary = evaluate_run_whole_idr(run_dir, testset, targets, target_dataset_idr)
        all_run_summaries.append(run_summary)
        all_scale_summaries.append(scale_summary)
    pd.concat(all_run_summaries).to_csv(output_summary_filepath, index=False)
    pd.concat(all_scale_summaries).to_csv(output_summary_filepath.replace('.csv', 'SCALES.csv'), index=False)


def collect_run_dirs(project_name, wandb_dir='/mnt/31dac31c-c4e2-4704-97bd-0788af37c5eb/shush/wandb/*/*'):
    wandb.login()
    api = wandb.Api()
    runs = api.runs(project_name)
    run_dirs = [glob.glob(wandb_dir+run.id)[0] for run in runs]
    return run_dirs

def collect_sweep_dirs(sweep_id, wandb_dir='/mnt/31dac31c-c4e2-4704-97bd-0788af37c5eb/shush/wandb/*/*'):
    api = wandb.Api()
    sweep = api.sweep(sweep_id)
    sweep_runs = sweep.runs
    run_dirs = [glob.glob(wandb_dir+run.id)[0] for run in sweep_runs]
    return run_dirs


if __name__ == '__main__':
    run_dirs = ['/mnt/31dac31c-c4e2-4704-97bd-0788af37c5eb/shush/wandb/wandb_elzar/run-20210923_162932-56p3xy2p',
                '/mnt/31dac31c-c4e2-4704-97bd-0788af37c5eb/shush/wandb/wandb_elzar/run-20210923_163101-7qjhy0ff',
                '/mnt/31dac31c-c4e2-4704-97bd-0788af37c5eb/shush/wandb/wandb_elzar/run-20210923_162940-pxy34wg8',
                '/mnt/31dac31c-c4e2-4704-97bd-0788af37c5eb/shush/wandb/wandb_elzar/run-20210923_162941-e3f2p92u',
                '/mnt/31dac31c-c4e2-4704-97bd-0788af37c5eb/shush/wandb/wandb_elzar/run-20210923_162937-r3jvc9kj']
    output_dir = 'summary_metrics_tables' # output dir
    util.make_dir(output_dir)
    wandb_project_name = 'BASENJI_BIN_LOSS_256' # project name in wandb
    csv_filename = wandb_project_name + '.csv'
    result_path = os.path.join(output_dir, csv_filename)
    testset, targets, target_dataset_idr = collect_datasets()
    if len(run_dirs) == 0:
        run_dirs = collect_run_dirs(wandb_project_name)
        print(run_dirs)
    process_run_list(run_dirs, result_path)
