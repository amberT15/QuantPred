import tensorflow as tf
import h5py
import explain
import custom_fit
import modelzoo
import os,json
import util
import time
import pandas as pd
import sys
import numpy as np
import tfr_evaluate, test_to_bw_fast

def main():

    augm_results = pd.read_csv('results_tfr_evaluate/augmentation_basenji.csv')
    all_run_metadata = []
    for run_path in glob.glob('paper_runs/basenji/augmentation_basenji/*'):
        all_run_metadata.append(get_run_metadata(run_path))
    all_run_metadata = pd.concat(all_run_metadata)
    all_run_metadata['dataset'] = ['random_chop' if 'random' in data_dir else 'peak_centered' for data_dir in all_run_metadata['data_dir'].values]
    model_paths = []
    for i, df in (all_run_metadata[all_run_metadata['bin_size']==128]).groupby(['crop', 'rev_comp', 'dataset']):
        assert df.shape[0] == 3, 'mip'
        model_paths.append(df.iloc[0]['run_dir'])

    testset_path = '/mnt/1a18a49e-9a31-4dbf-accd-3fb8abbfab2d/shush/4grid_atac/complete/peak_centered/i_3072_w_1/'
    save_performance = False

    base_directory = util.make_dir('robustness_test_results')
    testset, targets, stats = test_to_bw_fast.read_dataset(testset_path, True)

    # overlapping_testset_path = 'datasets/step1K_chr8_whole/i_3072_w_1/'

    for model_path in model_paths:
        start_time = time.time()
        print(model_path)
        output_directory = util.make_dir(os.path.join(base_directory, model_path.split('/')[-1]))
        # bw_filepath_suffix = '_pred.bw'
        variance_dataset_path = os.path.join(output_directory, 'variance_of_preds.h5')
        performance_result_path = os.path.join(output_directory, 'performance.csv')

        # load model and datasets
        model = modelzoo.load_model(model_path, compile=True)

        # compute variance of predictions and avg predictions
        preds, pred_vars, coords, Y = explain.batch_pred_robustness_test(testset, stats,
                                                                        model,
                                                                        batch_size=1,
                                                                        # visualize=False,
                                                                        shift_num=10)
        # save variance as h5
        h5_dataset = h5py.File(variance_dataset_path, 'w')
        h5_dataset.create_dataset('prediction_variance', data=pred_vars)
        h5_dataset.close()

        if save_performance=='save_performance':
            # save performance with metadata
            performance = tfr_evaluate.get_performance(Y, preds, targets, 'whole')
            metadata = tfr_evaluate.get_run_metadata(model_path)
            metadata_broadcasted = pd.DataFrame(np.repeat(metadata.values, performance.shape[0], axis=0), columns=metadata.columns)
            complete_dataset = pd.concat([performance, metadata_broadcasted], axis=1)
            complete_dataset.to_csv(performance_result_path)

        print('OVERALL TIME: '+ str((time.time()-start_time)//60))

def write_predictions_to_bw(preds, bw_path, cell_line=0,
                            chrom_size_path="/home/shush/genomes/GRCh38_EBV.chrom.sizes.tsv"):
    """create and write bw file with robust predictions"""
    opened_bw = test_to_bw_fast.open_bw(bw_path, chrom_size_path)
    for i in range(preds.shape[0]):
        add_to_bw(preds[i,:,cell_line], coords[i], opened_bw)
    opened_bw.close()

def add_to_bw(value, coord, bw_file_object, step=1):
    '''Write tf coord, values to open bw file'''
    chrom, start, end = coord
    bw_file_object.addEntries(chrom, int(start), values=value, span=step,
                              step=step)


# __main__
################################################################################
if __name__ == '__main__':
    main()
