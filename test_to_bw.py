#!/usr/bin/env python
import util
import os
import numpy as np
import csv
import pyBigWig
import tensorflow as tf
from modelzoo import GELU
import metrics
import loss
import custom_fit
import time
from loss import *
import yaml
import pandas as pd



def read_dataset(data_path):
    # data_path = 'datasets/only_test/complete/random_chop/i_2048_w_1'
    sts = util.load_stats(data_path)
    testset = util.make_dataset(data_path, 'test', sts, coords=True, shuffle=False)
    targets_path = os.path.join(data_path, 'targets.txt')
    targets = pd.read_csv(targets_path, delimiter='\t')['identifier'].values
    return testset, targets


def read_model(run_path):
    config_file = os.path.join(run_path, 'files', 'config.yaml')
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    loss_fn_str = config['loss_fn']['value']
    bin_size = config['bin_size']['value']
    loss_fn = eval(loss_fn_str)()
    model_path = os.path.join(run_path, 'files', 'best_model.h5')
    trained_model = tf.keras.models.load_model(model_path, custom_objects={"GELU": GELU})
    trained_model.compile(optimizer="Adam", loss=loss_fn)
    return trained_model, bin_size
#
def read_chrom_size(chrom_size_path):
    chrom_size = {}
    with open(chrom_size_path) as fd:
        rd = csv.reader(fd, delimiter="\t", quotechar='"')
        for line in rd:
            chrom_size[line[0]]=int(line[1])
    return chrom_size


def open_bw(bw_filename, chrom_size_path, output_dir='.'):
    chrom_sizes = read_chrom_size(chrom_size_path)
    bw = pyBigWig.open(os.path.join(output_dir, bw_filename), "w")
    bw.addHeader([(k, v) for k, v in chrom_sizes.items()], maxZooms=0)
    return bw


def make_true_pred_bw(testset, trained_model, targets, cell_line, bin_size,
              chrom_size_path, output_dir):

    cell_line_name = targets[cell_line]
    true_bw_filename = os.path.join(output_dir, cell_line_name+"_true.bw")
    pred_bw_filename = os.path.join(output_dir, cell_line_name+"_pred.bw")
    bed_filename = os.path.join(output_dir, cell_line_name+"_true.bed")
    bedfile = open(bed_filename, "w")
    true_bw = open_bw(true_bw_filename, chrom_size_path, output_dir)
    pred_bw = open_bw(pred_bw_filename, chrom_size_path, output_dir)
    print('Processing cell line '+targets[cell_line])
    for C, X, Y in testset: #per batch
        C = [str(c).strip('b\'').strip('\'') for c in C.numpy()] # coordinates
        P = trained_model(X) # make batch predictions

        for i, pred in enumerate(P): # per batch element
            chrom, start, end = C[i].split('_') # get chr, start, end
            start = int(start)
            true_bw.addEntries(chrom, start,
                values=np.array(np.squeeze(Y[i,:,cell_line]), dtype='float64'),
                span=1, step=1) # ground truth
            pred_bw.addEntries(chrom, start,
                values=np.array(np.squeeze(pred[:,cell_line]), dtype='float64'),
                span=bin_size, step=bin_size) # predictions
            bedfile.write('{}\t{}\t{}\n'.format(chrom, start, end))
    true_bw.close()
    pred_bw.close()
    bedfile.close()



def dataset_to_bw(data_path, run_path, cell_line, out_prefix='testset_bws',
                  chrom_size_path="/home/shush/genomes/GRCh38_EBV.chrom.sizes.tsv"):
    testset, targets = read_dataset(data_path) # get dataset
    trained_model, bin_size = read_model(run_path) # get model
    output_dir = os.path.join(run_path, 'files', out_prefix) # save output in wandb folder
    util.make_dir(output_dir)
    make_true_pred_bw(testset, trained_model, targets, cell_line, bin_size,
                  chrom_size_path, output_dir)

def bw_from_ranges(in_bw_filename, in_bed_filename, out_bw_filename,
                   chrom_size_path, out_dir, bin_size=1):
    '''
    This function creates bw file from existing bw file but only from specific
    bed ranges provided in the bed file
    '''
    in_bw = pyBigWig.open(in_bw_filename)
    out_bw = open_bw(out_bw_filename, chrom_size_path, out_dir)
    for line in open(in_bed_filename):
        cols = line.strip().split()
        vals = in_bw.values(cols[0], int(cols[1]), int(cols[2]))
        vals = np.array(vals, dtype='float64')
        vals = vals.reshape(len(vals)//bin_size, bin_size).mean(axis=1)
        out_bw.addEntries(cols[0], int(cols[1]), values=vals, span=bin_size,
                          step=bin_size)
    in_bw.close()
    out_bw.close()


def bw_with_threshold(in_bw, threshold, out_bw, out_bed):
    '''
    This function filters a bw file based on a given threshold and outputs a bw
    file with filtered values only and a bed file with the regions selected
    '''
    pass
