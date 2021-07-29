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
import subprocess
import gzip
import pandas as pd

def change_filename(filepath, new_binningsize=None, new_thresholdmethod=None):

    filename = os.path.basename(filepath)
    directory = filepath.split(filename)[0]
    celline, bigwigtype, bin, threshold = filename.split('.bw')[0].split('_')

    if new_binningsize != None:
        bin = new_binningsize
    if new_thresholdmethod != None:
        threshold = new_thresholdmethod

    new_filename = '_'.join([celline, bigwigtype, bin, threshold])+'.bw'
    return os.path.join(directory, new_filename)


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


def open_bw(bw_filename, chrom_size_path):
    chrom_sizes = read_chrom_size(chrom_size_path)
    # bw_filepath = os.path.join(output_dir, bw_filename)
    bw = pyBigWig.open(bw_filename, "w")
    bw.addHeader([(k, v) for k, v in chrom_sizes.items()], maxZooms=0)
    return bw


def make_truth_pred_bws(truth_bw_filename_suffix, pred_bw_filename_suffix, bed_filename_suffix,
                      testset, trained_model, bin_size, cell_line_names,
                      chrom_size_path, run_dir):

    bedfiles = {}
    pred_bws = {}
    truth_bws = {}
    cell_line_N = len(cell_line_names)
    for cell_line, cell_line_name in enumerate(cell_line_names):
        output_dir = util.make_dir(os.path.join(run_dir, str(cell_line) + '_' + cell_line_name))
        pred_bw_filename = os.path.join(output_dir, cell_line_name + pred_bw_filename_suffix)
        pred_bws[cell_line] = open_bw(pred_bw_filename, chrom_size_path)
        truth_bw_filename = os.path.join(output_dir, cell_line_name + truth_bw_filename_suffix)
        truth_bws[cell_line] = open_bw(truth_bw_filename, chrom_size_path)
        bed_filename = os.path.join(output_dir, cell_line_name + bed_filename_suffix)
        bedfiles[cell_line] = open(bed_filename, "w")
    for C, X, Y in testset: #per batch
        C = [str(c).strip('b\'').strip('\'') for c in C.numpy()] # coordinates
        P = trained_model(X) # make batch predictions

        for i, pred in enumerate(P): # per batch element
            chrom, start, end = C[i].split('_') # get chr, start, end
            start = int(start)
            for cell_line in range(cell_line_N):
                truth_bws[cell_line].addEntries(chrom, start,
                    values=np.array(np.squeeze(Y[i,:,cell_line]), dtype='float64'),
                    span=1, step=1) # ground truth
                pred_bws[cell_line].addEntries(chrom, start,
                    values=np.array(np.squeeze(pred[:,cell_line]), dtype='float64'),
                    span=bin_size, step=bin_size) # predictions
                bedfiles[cell_line].write('{}\t{}\t{}\n'.format(chrom, start, end))
    for cell_line in range(cell_line_N):
        truth_bws[cell_line].close()
        pred_bws[cell_line].close()
        bedfiles[cell_line].close()

def get_replicates(cell_line_name, repl_labels = ['r2', 'r12'], basenji_samplefiles=['/mnt/906427d6-fddf-41bf-9ec6-c3d0c37e766f/amber/ATAC/basenji_sample_r2_file.tsv', '/mnt/906427d6-fddf-41bf-9ec6-c3d0c37e766f/amber/ATAC/basenji_sample_r1,2_file.tsv']):
    replicate_filepaths = {}
    for b, basenji_samplefile in enumerate(basenji_samplefiles):
        basenji_samplefile_df = pd.read_csv(basenji_samplefile, sep='\t')
        cell_row = basenji_samplefile_df[basenji_samplefile_df['identifier']==cell_line_name]['file']
        assert not(len(cell_row) > 1), 'Multiple cell lines detected!'
        if len(cell_row) == 1:
            replicate_filepaths[repl_labels[b]] = cell_row.values[0]
    return replicate_filepaths


def get_idr(cell_line_name, idr_filename, basset_samplefile='/mnt/906427d6-fddf-41bf-9ec6-c3d0c37e766f/amber/ATAC/basset_sample_file.tsv'):

    basset_samplefile_df=pd.read_csv(basset_samplefile, sep='\t', header=None)
    idr_file_gz = basset_samplefile_df[basset_samplefile_df[0]==cell_line_name][1].values[0]
    bashCmd = "scp {} temp.bed.gz; gunzip temp.bed.gz; grep chr8 temp.bed|awk '{{print $1, $2, $3}}'|sort -k1,1 -k2,2n|uniq > {}; rm temp.bed".format(idr_file_gz, idr_filename)
    process = subprocess.Popen(bashCmd, shell=True)
    output, error = process.communicate()



def bw_from_ranges(in_bw_filename, in_bed_filename, out_bw_filename,
                   chrom_size_path, bin_size=1, threshold=-1,
                   out_bed_filename=''):
    '''
    This function creates bw file from existing bw file but only from specific
    bed ranges provided in the bed file, and optionally thresholds the bed file
    as well as optionally outputs the regions selected if out_bed_filename provided
    '''
    if len(out_bed_filename) > 0:
        # bedfile_path = os.path.join(out_dir, out_bed_filename)
        bedfile = open(out_bed_filename, "w")
    in_bw = pyBigWig.open(in_bw_filename)
    out_bw = open_bw(out_bw_filename, chrom_size_path)
    in_bedfile = open(in_bed_filename)
    for line in in_bedfile:
        # print(line)
        cols = line.strip().split()
        vals = in_bw.values(cols[0], int(cols[1]), int(cols[2]))
        vals = np.array(vals, dtype='float64')

        # if np.sum(np.isnan(vals))>0:
        #     print(in_bw_filename, in_bed_filename)
        #     print(cols)
        if np.max(vals) > threshold:
            vals = vals.reshape(len(vals)//bin_size, bin_size).mean(axis=1)
            out_bw.addEntries(cols[0], int(cols[1]), values=vals, span=bin_size,
                              step=bin_size)
            if len(out_bed_filename) > 0:
                bedfile.write(line)

    in_bw.close()
    out_bw.close()
    if len(out_bed_filename) > 0:
        bedfile.close()

def process_cell_line(run_path,
                      threshold=2,
                      data_path='datasets/chr8/complete/random_chop/i_2048_w_1',
                      chrom_size_path="/home/shush/genomes/GRCh38_EBV.chrom.sizes.tsv"):


    testset, targets = read_dataset(data_path) # get dataset
    trained_model, bin_size = read_model(run_path) # get model


    truth_bw_filename_suffix = "_truth_1_raw.bw"
    pred_bw_filename_suffix = "_pred_{}_raw.bw".format(bin_size)
    bed_filename_suffix = "_truth_1_raw.bed"
    run_subdir = util.make_dir(os.path.join(run_path, 'bigwigs'))

    make_truth_pred_bws(truth_bw_filename_suffix, pred_bw_filename_suffix, bed_filename_suffix,
                          testset, trained_model, bin_size, targets,
                          chrom_size_path, run_subdir)

    for subdir in os.listdir(run_subdir):


        output_dir = os.path.join(run_subdir, subdir)
        subdir_split = subdir.split('_')
        assert len(subdir_split) == 2, 'Check subdirectory names for underscores!'
        cell_line_index, cell_line_name = subdir_split
        bed_filename = os.path.join(output_dir, cell_line_name + '_truth_1_raw.bed')
        replicate_filepaths = get_replicates(cell_line_name)
        cell_line_truth_idr = os.path.join(output_dir, cell_line_name + '_truth_1_idr.bed')
        get_idr(cell_line_name, cell_line_truth_idr)
        print('Processing cell line '+cell_line_name)
        # make nonthresholded non binned replicates
        rX_bw_filenames = []
        for rX, rX_bw_path in replicate_filepaths.items():
            out_rX = os.path.join(output_dir, cell_line_name + '_{}_1_raw.bw'.format(rX))
            rX_bw_filenames.append(out_rX)
            bw_from_ranges(rX_bw_path, bed_filename, out_rX, chrom_size_path)
        # bin truth, rXs
        truth_bw_filename = os.path.join(output_dir, cell_line_name+truth_bw_filename_suffix)
        binned_filenames = []
        for in_bw in rX_bw_filenames+[truth_bw_filename]:
            out_bw = change_filename(in_bw, new_binningsize=str(bin_size))
            binned_filenames.append(out_bw)
            bw_from_ranges(in_bw, bed_filename, out_bw, chrom_size_path, bin_size=bin_size)

        pred_bw_filename = os.path.join(output_dir, cell_line_name+pred_bw_filename_suffix)
        # threshold all using IDR file of truth bw
        for binned_filename in binned_filenames+[pred_bw_filename]:
            out_bw = change_filename(binned_filename, new_thresholdmethod='idr')
            bw_from_ranges(binned_filename, cell_line_truth_idr, out_bw, chrom_size_path)
        print('truth_thresh_filename')
        # threshold all using IDR file of truth bw
        thresh_bedfile = truth_bw_filename.split('.bw')[0]+'_thresh{}.bed'.format(threshold)
        thresh_str = 'thresh'+str(threshold)
        truth_thresh_filename = change_filename(truth_bw_filename, new_thresholdmethod=thresh_str)
        print(truth_thresh_filename)
        bw_from_ranges(truth_bw_filename, bed_filename, truth_thresh_filename, chrom_size_path, threshold=threshold, out_bed_filename=thresh_bedfile)
        for binned_filename in binned_filenames+[pred_bw_filename]:
            out_thresh = change_filename(binned_filename, new_thresholdmethod=thresh_str)
            bw_from_ranges(binned_filename, thresh_bedfile, out_thresh, chrom_size_path)
