#!/usr/bin/env python
import util
import os, shutil
import numpy as np
import csv
import pyBigWig
import tensorflow as tf
from modelzoo import GELU
import metrics
import loss
import custom_fit
import time
from scipy import stats
from loss import *
import yaml
import subprocess
import gzip
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
import time

def enforce_constant_size(bed_path, output_path, window, compression=None):
    """generate a bed file where all peaks have same size centered on original peak"""

    # load bed file

    df = pd.read_csv(bed_path, sep=' ', header=None, compression=compression)

    df.columns = [0, 1, 2]
    chrom = df[0].to_numpy().astype(str)
    start = df[1].to_numpy()
    end = df[2].to_numpy()
    #print('# bed coordinates', len(end))
    # calculate center point and create dataframe
    middle = np.round((start + end)/2).astype(int)
    half_window = np.round(window/2).astype(int)

    # calculate new start and end points
    start = middle - half_window
    end = middle + half_window

    # filter any negative start positions
    data = {}
    for i in range(len(df.columns)):
        data[i] = df[i].to_numpy()
    data[1] = start
    data[2] = end
    #print('# coordinates after removing negatives', len(end))
    # create new dataframe
    df_new = pd.DataFrame(data);
#     print(df_new[df_new[1]<0])
    df_new = df_new[df_new.iloc[:,1] > 0]
    df_new = df_new[df_new.iloc[:,2] > 0]
    # save dataframe with fixed width window size to a bed file
    df_new.to_csv(output_path, sep='\t', header=None, index=False)

def change_filename(filepath, new_binningsize=None, new_thresholdmethod=None):
    '''This funciton switches between filenames used for bw files'''
    filename = os.path.basename(filepath) # extract file name from path
    directory = filepath.split(filename)[0] # extract folder name
    # split filename into variables
    celline, bigwigtype, bin, threshold = filename.split('.bw')[0].split('_')
    if new_binningsize != None: # if new bin size provided replace
        bin = new_binningsize
    if new_thresholdmethod != None: # if new threshold provided replace
        threshold = new_thresholdmethod
    # construct new filename
    new_filename = '_'.join([celline, bigwigtype, bin, threshold])+'.bw'
    return os.path.join(directory, new_filename) # return full path

def read_dataset(data_path):
    '''This function returns testset and corresponding cell lines'''
    # data_path = 'datasets/only_test/complete/random_chop/i_2048_w_1' - test set
    sts = util.load_stats(data_path) # load stats file
    # make dataset from tfrecords
    testset = util.make_dataset(data_path, 'test', sts, coords=True, shuffle=False)
    targets_path = os.path.join(data_path, 'targets.txt') # load cell line names
    targets = pd.read_csv(targets_path, delimiter='\t')['identifier'].values
    return testset, targets # return test set and cell line names

def get_config(run_path):
    '''This function returns config of a wandb run as a dictionary'''
    config_file = os.path.join(run_path, 'files', 'config.yaml')
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def read_model(run_path):
    '''This function loads a per-trained model'''
    config = get_config(run_path) # load wandb config
    loss_fn_str = config['loss_fn']['value'] # get loss
    bin_size = config['bin_size']['value'] # get bin size
    loss_fn = eval(loss_fn_str)() # turn loss into function
    model_path = os.path.join(run_path, 'files', 'best_model.h5') # pretrained model
    # load model
    trained_model = tf.keras.models.load_model(model_path, custom_objects={"GELU": GELU})
    trained_model.compile(optimizer="Adam", loss=loss_fn)
    return trained_model, bin_size # model and bin size

def read_chrom_size(chrom_size_path):
    '''Load chromosome size file'''
    chrom_size = {}
    with open(chrom_size_path) as fd:
        rd = csv.reader(fd, delimiter="\t", quotechar='"')
        for line in rd:
            chrom_size[line[0]]=int(line[1])
    return chrom_size

def open_bw(bw_filename, chrom_size_path):
    '''This function opens a new bw file'''
    assert not os.path.isfile(bw_filename), 'Bw at {} alread exists!'.format(bw_filename)
    chrom_sizes = read_chrom_size(chrom_size_path) # load chromosome sizes
    bw = pyBigWig.open(bw_filename, "w") # open bw
    bw.addHeader([(k, v) for k, v in chrom_sizes.items()], maxZooms=0)
    return bw # bw file

def get_mean_per_range(bw_path, bed_path):
    '''This function reads bw (specific ranges of bed file) into numpy array'''
    bw = pyBigWig.open(bw_path)
    bw_list = []
    for line in open(bed_path):
        cols = line.strip().split()
        vals = bw.values(cols[0], int(cols[1]), int(cols[2]))
        bw_list.append(vals)
    bw.close()
    return bw_list

def remove_nans(all_vals_dict):
    '''This function masks nans in all values in a dict'''
    for i,(k, v) in enumerate(all_vals_dict.items()): # for each k, v
        if np.isnan(v).sum()>0: # if any nans present
            if 'nan_mask' not in locals(): # if nan_mask variable not created
                nan_mask = ~(np.isnan(v)) # make a variable nan_mask
            else:  # if already present
                nan_mask *= ~(np.isnan(v)) # add to the existing nan mask
    nonan_dict = {} # clean dictionary
    for k,v in all_vals_dict.items(): # for each k, v in original dict
        if 'nan_mask' in locals(): # if nans were found in any v
            nonan_dict[k] = v[nan_mask] # return the original masking out nans
        else: # if no nans were present
            nonan_dict[k] = v # just return original values
    return nonan_dict # return filtered dict of values

def make_truth_pred_bws(truth_bw_filename_suffix, pred_bw_filename_suffix,
                        bed_filename_suffix, testset, trained_model, bin_size,
                        cell_line_names, chrom_size_path, run_dir):
    '''This function makes ground truth and prediction bw-s from tfrecords dataset'''
    # open bw and bed files
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
    # go through test set data points
    for C, X, Y in testset: #per batch
        C = [str(c).strip('b\'').strip('\'') for c in C.numpy()] # coordinates
        P = trained_model(X) # make batch predictions
        for i, pred in enumerate(P): # per batch element
            chrom, start, end = C[i].split('_') # get chr, start, end
            start = int(start) # to feed into bw making function
            for cell_line in range(cell_line_N): # per cell line
                # write to ground truth file
                truth_bws[cell_line].addEntries(chrom, start,
                    values=np.array(np.squeeze(Y[i,:,cell_line]), dtype='float64'),
                    span=1, step=1)
                # write to prediction bw file
                pred_bws[cell_line].addEntries(chrom, start,
                    values=np.array(np.squeeze(pred[:,cell_line]), dtype='float64'),
                    span=bin_size, step=bin_size)
                # write ti bedfile (same for each cell line but needed for later)
                bedfiles[cell_line].write('{}\t{}\t{}\n'.format(chrom, start, end))
    # close everything
    for cell_line in range(cell_line_N):
        truth_bws[cell_line].close()
        pred_bws[cell_line].close()
        bedfiles[cell_line].close()

def merge_bed(in_bed_filename):
    '''This function merges bed consequtive bed ranges'''
    split_filename = in_bed_filename.split('/') # deconstruct file name
    # rejoin file name with prefic 'merge'
    in_bed_filename_merged = '/'.join(split_filename[:-1] + ['merged_' + split_filename[-1]])
    # str command line for bedtools merge
    bashCmd = 'bedtools merge -i {} > {}'.format(in_bed_filename, in_bed_filename_merged)
    process = subprocess.Popen(bashCmd, shell=True)
    output, error = process.communicate()
    return in_bed_filename_merged # return new filename

def get_list_pr(list1, list2):
    '''This function flattens np arrays and computes pearson r'''
    pr = stats.pearsonr(np.concatenate(list1), np.concatenate(list2))[0]
    assert ~np.isnan(pr)
    return pr

def scipy_get_pr(bw_paths, bedfile='/home/shush/profile/QuantPred/bin_exp/truth/merged_truth_1_raw.bed'):
    '''This function computes pearson r from two bigwig files'''
    all_vals_dict_nans = {} # dictionary of all values
    for bw_path in bw_paths: # for each bw
        vals = get_mean_per_range(bw_path, bedfile) # convert bw to list of vals
        # convert to flattened np array
        all_vals_dict_nans[bw_path] = np.array([v  for v_sub in vals for v in v_sub])
        # remove nans
        all_vals_dict_1d = remove_nans(all_vals_dict_nans)
    # make sure there's enough values left
    assert len(all_vals_dict_1d[bw_paths[0]])>1 and len(all_vals_dict_1d[bw_paths[1]])>1, bw_paths
    # get pearson r
    pr = stats.pearsonr(all_vals_dict_1d[bw_paths[0]], all_vals_dict_1d[bw_paths[1]])[0]
    assert ~np.isnan(pr), 'Pearson R is nan for these {}'.format(bw_paths)
    return pr

def get_replicates(cell_line_name, repl_labels = ['r2', 'r12'],
                    basenji_samplefiles=['/mnt/906427d6-fddf-41bf-9ec6-c3d0c37e766f/amber/ATAC/basenji_sample_r2_file.tsv', '/mnt/906427d6-fddf-41bf-9ec6-c3d0c37e766f/amber/ATAC/basenji_sample_r1,2_file.tsv']):
    '''This function finds paths to replicates for a specific cell line'''
    replicate_filepaths = {} #filename dictionary
    # per samplefile of replicates
    for b, basenji_samplefile in enumerate(basenji_samplefiles):
        # read in the samplefile
        basenji_samplefile_df = pd.read_csv(basenji_samplefile, sep='\t')
        # get the row with cell line name
        cell_row = basenji_samplefile_df[basenji_samplefile_df['identifier']==cell_line_name]['file']
        # make sure no duplicates detected
        assert not(len(cell_row) > 1), 'Multiple cell lines detected!'
        if len(cell_row) == 1: # if cell line replicate found
            replicate_filepaths[repl_labels[b]] = cell_row.values[0] # get filepath
    return replicate_filepaths # return dict of repl type and path

def get_idr(cell_line_name, idr_filename,
            basset_samplefile='/mnt/906427d6-fddf-41bf-9ec6-c3d0c37e766f/amber/ATAC/basset_sample_file.tsv',
           range_size = 2048,
           unmap_bed='/home/shush/genomes/GRCh38_unmap.bed'):
    '''This function makes cell line specific IDR file with constant window size for test set'''

    # make bed filename to be used in pearson r calculation with no merging of ranges
    split_filename = idr_filename.split('/')
    window_enf_idr = '/'.join(split_filename[:-1]+[split_filename[-1].split('.bed')[0]+ '_const' +'.bed'])
    nan_window_enf_idr = window_enf_idr + 'nan'
    # read in IDR samplefile
    basset_samplefile_df=pd.read_csv(basset_samplefile, sep='\t', header=None)
    # find cell line specific IDR file
    idr_file_gz = basset_samplefile_df[basset_samplefile_df[0]==cell_line_name][1].values[0]
    # str of command line command to filter test set peaks for cell line into new bed file
    interm_bed = '{}_idr_strict_peaks.bed'.format(cell_line_name)
    make_bedfile = "scp {} temp.bed.gz; gunzip temp.bed.gz; grep chr8 temp.bed|awk '{{print $1, $2, $3}}'|sort -k1,1 -k2,2n|uniq > {}; rm temp.bed".format(idr_file_gz, interm_bed)
    process = subprocess.Popen(make_bedfile, shell=True)
    output, error = process.communicate()
    # make new bedfile with constant bed ranges
    enforce_constant_size(interm_bed, nan_window_enf_idr, range_size)
    # remove regions partially in unmap regions
    filter_bed = 'bedtools intersect -v -a {} -b {} > {}'.format(nan_window_enf_idr, unmap_bed, window_enf_idr)
    process = subprocess.Popen(filter_bed, shell=True)
    output, error = process.communicate()
    # merge ranges so that bw writing can happen later
    merge_bed = 'bedtools merge -i {} > {}; rm {}'.format(window_enf_idr, idr_filename, interm_bed)
    process = subprocess.Popen(merge_bed, shell=True)
    output, error = process.communicate()



def bw_from_ranges(in_bw_filename, in_bed_filename, out_bw_filename,
                   chrom_size_path, bin_size=1, threshold=-1,
                   out_bed_filename=''):
    '''
    This function creates bw file from existing bw file but only from specific
    bed ranges provided in the bed file, and optionally thresholds the bed file
    as well as optionally outputs the regions selected if out_bed_filename provided
    '''
    if len(out_bed_filename) > 0: # if out_bed_filename given to save recorded ranges
        bedfile = open(out_bed_filename, "w") # open new bed file
    in_bw = pyBigWig.open(in_bw_filename) # open existing bw
    out_bw = open_bw(out_bw_filename, chrom_size_path) # open new bw
    in_bedfile = open(in_bed_filename) # open existing bed file
    for line in in_bedfile: # per bed range
        cols = line.strip().split()
        vals = in_bw.values(cols[0], int(cols[1]), int(cols[2])) # get coords
        vals = np.array(vals, dtype='float64') # get values
        if np.max(vals) > threshold: # if above threshold
            # write values to new bw using bin size as step
            vals = vals.reshape(len(vals)//bin_size, bin_size).mean(axis=1)
            out_bw.addEntries(cols[0], int(cols[1]), values=vals, span=bin_size,
                              step=bin_size)
            if len(out_bed_filename) > 0: # if bed file of ranges needed
                bedfile.write(line) # record range above threshold
    # close files
    in_bw.close()
    out_bw.close()
    if len(out_bed_filename) > 0:
        bedfile.close()

def process_run(run_path,
                      threshold=2,
                      data_path='datasets/chr8/complete/random_chop/i_2048_w_1',
                      chrom_size_path="/home/shush/genomes/GRCh38_EBV.chrom.sizes.tsv",
                      get_replicates=False,
                      bigwig_foldername='bigwigs'):
    '''This function processes a wandb run and outputs bws of following types:
    - ground truth base resolution
    - ground truth binned if model is trained on binned dataset
    - prediction binned at whatever model is trained on
    - ground truth and prediction thresholded using provided threshold (optional)
    - ground truth and pred of IDR peaks per cell line (cut from full bw not
      predicted anew) (optional)
    - ground truth of replicates in every filtering type (optional)'''
    testset, targets = read_dataset(data_path) # get dataset
    trained_model, bin_size = read_model(run_path) # get model
    # set filename for base res ground truth
    truth_bw_filename_suffix = "_truth_1_raw.bw"
    # set filename for prediction bw at model resolution
    pred_bw_filename_suffix = "_pred_{}_raw.bw".format(bin_size)
    bed_filename_suffix = "_truth_1_raw.bed" # filename for bed file of ranges
    # new subfolder to save bigwigs in
    run_subdir = util.make_dir(os.path.join(run_path, bigwig_foldername))
    # make ground truth, pred bigwigs and bed file of ranges where dataset is
    # for each cell line in a separate subdir in run_subdir
    print('Making ground truth and prediction bigwigs')
    t0 = time.time()
    make_truth_pred_bws(truth_bw_filename_suffix, pred_bw_filename_suffix, bed_filename_suffix,
                          testset, trained_model, bin_size, targets,
                          chrom_size_path, run_subdir)
    t1 = time.time()
    print('Time = {}mins'.format((t1-t0)//60))
    for subdir in tqdm(os.listdir(run_subdir)): # per cell line directory
        print(subdir)
        output_dir = os.path.join(run_subdir, subdir) # cell line full path
        subdir_split = subdir.split('_') # split into id and cell line name
        # make sure no other file is detected
        assert len(subdir_split) == 2, 'Check subdirectory names for underscores!'
        cell_line_index, cell_line_name = subdir_split # read in id and name
        # define base res ground truth path
        bed_filename = os.path.join(output_dir, cell_line_name + '_truth_1_raw.bed')
        # define IDR bw path
        cell_line_truth_idr = os.path.join(output_dir, cell_line_name + '_truth_1_idr.bed')
        get_idr(cell_line_name, cell_line_truth_idr) # find cell line IDR bed file
        print('Processing cell line '+cell_line_name)
        #### make nonthresholded non binned replicates
        rX_bw_filenames = []
        if get_replicates: # if replicates needed
            replicate_filepaths = get_replicates(cell_line_name) # find em
            for rX, rX_bw_path in replicate_filepaths.items(): # per replicate
                # make a new base res bigwig path
                out_rX = os.path.join(output_dir, cell_line_name + '_{}_1_raw.bw'.format(rX))
                # save bw filename for later binning and thresholding
                rX_bw_filenames.append(out_rX)
                # extract bw values same as the base res ground truth bw
                bw_from_ranges(rX_bw_path, bed_filename, out_rX, chrom_size_path)
        #### bin ground truth, replicates (rXs = r2, r12, etc.)
        truth_bw_filename = os.path.join(output_dir, cell_line_name+truth_bw_filename_suffix)
        # new binned filenames which may be original ones if bin size = 1 or
        # new one if bin size more than 1
        binned_filenames = []
        # for each newly made bw except for prediction (which is already binned
        # to whatever we need)
        for in_bw in rX_bw_filenames+[truth_bw_filename]:
            # make new filename
            out_bw = change_filename(in_bw, new_binningsize=str(bin_size))
            binned_filenames.append(out_bw) # save for later
            if bin_size != 1: # if bin less than one don't redo bw making!
                bw_from_ranges(in_bw, bed_filename, out_bw, chrom_size_path, bin_size=bin_size)
        # add pred to the binned bw filename set
        pred_bw_filename = os.path.join(output_dir, cell_line_name+pred_bw_filename_suffix)
        print(binned_filenames+[pred_bw_filename])
        #### threshold all using IDR file of ground truth bw
        for binned_filename in binned_filenames+[pred_bw_filename]: # for all new bws
            # make IDR bw
            out_bw = change_filename(binned_filename, new_thresholdmethod='idr')
            # filter IDR peak regions only
            bw_from_ranges(binned_filename, cell_line_truth_idr, out_bw, chrom_size_path)

        # if threhsold given, threshold all using absolute threshold
        if threshold > 0:
            # new bed filename
            thresh_str = 'thresh'+str(threshold)
            thresh_bedfile = truth_bw_filename.split('.bw')[0]+'_{}.bed'.format(thresh_str)
            # new bw filename for ground truth
            truth_thresh_filename = change_filename(truth_bw_filename, new_thresholdmethod=thresh_str)
            bw_from_ranges(truth_bw_filename, bed_filename, truth_thresh_filename, chrom_size_path, threshold=threshold, out_bed_filename=thresh_bedfile)
            # for all binned bws that are to be thresholded
            for binned_filename in binned_filenames+[pred_bw_filename]:
                print(binned_filename)
                out_thresh = change_filename(binned_filename, new_thresholdmethod=thresh_str)
                if 'truth_1_thresh' not in out_thresh: # this one would already be made above
                    bw_from_ranges(binned_filename, thresh_bedfile, out_thresh, chrom_size_path)
