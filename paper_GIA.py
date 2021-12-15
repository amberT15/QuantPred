import sys
import tensorflow as tf
import h5py, os, yaml
import umap.umap_ as umap
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import re
import seaborn as sns
from scipy import stats
import pandas as pd
import subprocess
from scipy.stats import pearsonr
from tqdm import tqdm
import glob
from scipy.stats import mannwhitneyu
from statannotations.Annotator import Annotator
import tfr_evaluate, util
from test_to_bw_fast import read_model, get_config
import explain
import embed
import metrics
import quant_GIA
from optparse import OptionParser



def main():
    usage = 'usage: %prog [options] <motifs> <cell_line>'
    parser = OptionParser(usage)
    (options, args) = parser.parse_args()

    if len(args) != 2:
      parser.error('Must provide motifs and cell line.')
    else:
      motifs = args[0].split(',')
      cell_line = int(args[1])

    print('Processing')
    print(motifs)
    # load and get model layer
    run_path = 'paper_runs/new_models/32_res/run-20211023_095131-w6okxt01'
    layer = -3
    model, bin_size = read_model(run_path, compile_model=False)
    aux_model = tf.keras.Model(inputs=model.inputs, outputs=model.layers[layer].output)
    output_dir = util.make_dir('paper_GIA_csvs')
    # load and threshold data
    testset, targets = tfr_evaluate.collect_whole_testset(coords=True)
    C, X, Y = util.convert_tfr_to_np(testset, 3)
    for testset_type in ['all threshold', 'cell line low coverage', 'cell line high coverage']:
        print(testset_type)
        selected_X = select_set(testset_type, C, X, Y, cell_line=cell_line)
        gi = quant_GIA.GlobalImportance(model, targets)
        gi.occlude_all_motif_instances(selected_X, motifs, func='mean')
        df = gi.summary_remove_motifs[0]
        file_prefix = '{}_in_{}_{}'.format(df['motif pattern'].values[0], targets[cell_line], testset_type)
        df.to_csv(os.path.join(output_dir, file_prefix+'csv'), index=None)


def select_set(testset_type, C, X, Y, cell_line=None):
    if testset_type == 'whole':
        return (X)
    elif testset_type == 'all threshold':
        threshold_mask = (Y.max(axis=1)>2).any(axis=-1)
        return (X[threshold_mask])
    elif testset_type == 'cell line high coverage':
        assert cell_line, 'No cell line provided!'
        _, thresh_X, _ = embed.threshold_cell_line_np(C, X, Y, cell_line,
                                                    more_than=2,
                                                    less_than=None)
        return (thresh_X)
    elif testset_type == 'cell line low coverage':
        assert cell_line, 'No cell line provided!'
        _, thresh_X, _ = embed.threshold_cell_line_np(C, X, Y, cell_line,
                                                    more_than=1,
                                                    less_than=2)
        return (thresh_X)
    else:
        print('Wrong please try again thank you bye')
        exit()


################################################################################
if __name__ == '__main__':
  main()
