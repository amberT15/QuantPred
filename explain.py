import numpy as np
import tensorflow as tf
from tensorflow import keras
import math
import matplotlib.pyplot as plt
import pandas as pd
import logomaker
import subprocess
import os, shutil, h5py

def plot_saliency(saliency_map):

    fig, axs = plt.subplots(saliency_map.shape[0],1,figsize=(200,5*saliency_map.shape[0]))
    for n, w in enumerate(saliency_map):
        ax = axs[n]
        #plot saliency map representation
        saliency_df = pd.DataFrame(w.numpy(), columns = ['A','C','G','T'])
        logomaker.Logo(saliency_df, ax=ax)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('none')
        ax.xaxis.set_ticks_position('none')
        plt.xticks([])
        plt.yticks([])
    return plt


def select_top_pred(pred,num_task,top_num):

    task_top_list = []
    for i in range(0,num_task):
        task_profile = pred[:,:,i]
        task_mean =np.squeeze(np.mean(task_profile,axis = 1))
        task_index = task_mean.argsort()[-top_num:]
        task_top_list.append(task_index)
    task_top_list = np.array(task_top_list)
    return task_top_list

def vcf_test(ref,alt,coords,model,background_size = 100):
    # score for reference and alternative allele
    ref_pred = model.predict(ref)
    alt_pred = model.predict(alt)
    ref_pred_cov = np.sum(ref_pred,axis = (1,2))
    alt_pred_cov = np.sum(alt_pred,axis = (1,2))

    d = {'chromosome': coords[:,0], 'start': coords[:,1],'end':coords[:,2]}
    df = pd.DataFrame(data=d)

    df['ref'] = ref_pred_cov
    df['alt'] = alt_pred_cov

    #creating random background mutations
    #mutations will be at least 100 nt apart from SNP position
    background_distribution = []
    for i,ref_seq in enumerate(ref):
        mut_loci = np.random.randint(100,923,size = background_size)
        direction = np.random.choice([-1,1])
        mut_loci = len(ref_seq)/2 + mut_loci * direction
        mut_loci = mut_loci.astype('int')
        mut_batch = np.tile(ref_seq,(background_size,1,1))
        mut_batch[range(0,100),mut_loci] = [0,0,0,0]
        mut_base = np.random.randint(0,4,size = background_size)
        mut_batch[range(0,100),mut_loci,mut_base] = 1

        mut_pred = model.predict(mut_batch)
        mut_pred_cov = np.sum(mut_pred,axis = (1,2))
        background_distribution.append(mut_pred_cov)
    df['background'] = background_distribution
    return df

def complete_saliency(X,model,class_index,func = tf.math.reduce_mean):
  """fast function to generate saliency maps"""
  if not tf.is_tensor(X):
    X = tf.Variable(X)

  with tf.GradientTape() as tape:
    tape.watch(X)
    if class_index is not None:
      outputs = func(model(X)[:,:,class_index])
    else:
      raise ValueError('class index must be provided')
  return tape.gradient(outputs, X)


def peak_saliency_map(X, model, class_index,window_size,func=tf.math.reduce_mean):
    """fast function to generate saliency maps"""
    if not tf.is_tensor(X):
        X = tf.Variable(X)

    with tf.GradientTape() as tape:
        tape.watch(X)
        pred = model(X)

        peak_index = tf.math.argmax(pred[:,:,class_index],axis=1)
        batch_indices = []

        if int(window_size) > 50:
            bin_num = 1
        elif int(window_size) == 32:
            bin_num = 3
        else:
            bin_num = 50

        for i in range(0,X.shape[0]):
            column_indices = tf.range(peak_index[i]-int(bin_num/2),peak_index[i]+math.ceil(bin_num/2),dtype='int32')
            row_indices = tf.keras.backend.repeat_elements(tf.constant([i]),bin_num, axis=0)
            full_indices = tf.stack([row_indices, column_indices], axis=1)
            batch_indices.append([full_indices])
            outputs = func(tf.gather_nd(pred[:,:,class_index],batch_indices),axis=2)

        return tape.gradient(outputs, X)

def fasta2list(fasta_file):
    fasta_coords = []
    seqs = []
    header = ''

    for line in open(fasta_file):
        if line[0] == '>':
            #header = line.split()[0][1:]
            fasta_coords.append(line[1:].rstrip())
        else:
            s = line.rstrip()
            s = s.upper()
            seqs.append(s)

    return fasta_coords, seqs



def dna_one_hot(seq):

    seq_len = len(seq)
    seq_start = 0
    seq = seq.upper()

    # map nt's to a matrix len(seq)x4 of 0's and 1's.
    seq_code = np.zeros((seq_len, 4), dtype='float16')

    for i in range(seq_len):
        if i >= seq_start and i - seq_start < len(seq):
            nt = seq[i - seq_start]
            if nt == 'A':
                seq_code[i, 0] = 1
            elif nt == 'C':
                seq_code[i, 1] = 1
            elif nt == 'G':
                seq_code[i, 2] = 1
            elif nt == 'T':
                seq_code[i, 3] = 1
            else:
                seq_code[i,:] = 0.25



    return seq_code

def enforce_const_range(site, window):
    half_window = np.round(window/2).astype(int)
    start = site - half_window
    end = site + half_window
    return start, end

def combine_beds(samplefile, out_path):
    bed_paths = pd.read_csv(samplefile, sep='\t', header=None)[1].values
    combined_csv = pd.concat([pd.read_csv(f, sep='\t', header=None) for f in bed_paths ])
    combined_csv.to_csv(out_path, sep='\t', header=None, index=None)

def filter_dataset(dsq_path, out_path='dsq_all.bed'):
    dsq_all = pd.read_csv(dsq_path, sep='\t')
    dsq_filt = dsq_all[(dsq_all['chrom']=='chr8')]
    dsq_filt[['a1','a2']] = dsq_filt['genotypes'].str.split('/',expand=True) # write into separate columns
    dsq_filt.to_csv(out_path, sep='\t', header=False, index=None)
    return list(dsq_filt)

def bed_intersect(dataset_bed, comb_peak, out_path):
    bashCmd = "bedtools intersect -a {} -b {} > {}".format(dataset_bed, comb_peak, out_path)
    process = subprocess.Popen(bashCmd, shell=True)
    output, error = process.communicate()
    print(error)

def extend_ranges(column_names, bedfile, out_path, window):
    dsq_df = pd.read_csv(bedfile, sep='\t', header=None, index_col=None)
    dsq_df.columns = column_names #list(dsq_filt)
    dsq_filt = dsq_df[['chrom', 'snpChromStart', 'snpChromEnd', 'a1', 'a2',
                        'strand', 'rsid', 'pred.fit.pctSig']]
    start, end = enforce_const_range(dsq_filt['snpChromEnd']-1, window)
    dsq_ext = dsq_filt.copy()
    dsq_ext.iloc[:,1] = start.values
    dsq_ext.iloc[:,2] = end.values
    dsq_nonneg = dsq_ext[dsq_ext['snpChromStart']>0]
    dsq_nonneg = dsq_nonneg.reset_index(drop=True)
    dsq_nonneg.to_csv(out_path, header=None, sep='\t', index=None)
    return dsq_nonneg

def bed_to_fa(bedfile='test_ds.csv', out_fa='test_ds.fa',
              genome_file='/home/shush/genomes/hg19.fa'):
    bashCmd = "bedtools getfasta -fi {} -bed {} -s -fo {}".format(genome_file, bedfile, out_fa)
    process = subprocess.Popen(bashCmd, shell=True)
    output, error = process.communicate()
    print(error)





def str_to_onehot(coords_list, seqs_list, dsq_nonneg, window):

    N = len(seqs_list)
    mid = window // 2
    onehot_ref = np.empty((N, window, 4))
    onehot_alt = np.empty((N, window, 4))
    coord_np = np.empty((N, 4)) # chrom, start, end coordinate array

    for i,(chr_s_e, seq) in enumerate(zip(coords_list, seqs_list)):
        alt = ''
        strand = chr_s_e.split('(')[-1].split(')')[0]
        pos_dict = {'+': mid, '-':mid-1}
        pos = pos_dict[strand]
        coord_np[i,3] = pos_dict[strand] - mid-1

        if seq[pos] == dsq_nonneg['a1'][i]:
            alt = dsq_nonneg['a2'][i]

        elif seq[pos] == dsq_nonneg['a2'][i]:
            alt = dsq_nonneg['a1'][i]
        else:
            break

        chrom, s_e = chr_s_e.split('(')[0].split(':')
        s, e = s_e.split('-')
        coord_np[i, :3] = int(chrom.split('chr')[-1]), int(s), int(e)

        onehot = dna_one_hot(seq)
        onehot_ref[i,:,:] = onehot

        onehot_alt[i, :, :] = onehot
        onehot_alt[i, mid, :] = dna_one_hot(alt)[0]

    return (onehot_ref, onehot_alt, coord_np)


def onehot_to_h5(onehot_ref, onehot_alt, coord_np, out_dir='.', filename='onehot.h5'):
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    onehot_ref_alt = h5py.File(os.path.join(out_dir, filename), 'w')
    onehot_ref_alt.create_dataset('ref', data=onehot_ref, dtype='float32')
    onehot_ref_alt.create_dataset('alt', data=onehot_alt, dtype='float32')
    onehot_ref_alt.create_dataset('fasta_coords', data=coord_np, dtype='i')
    onehot_ref_alt.close()

def table_to_h5(dsq_path,
                samplefile='/mnt/906427d6-fddf-41bf-9ec6-c3d0c37e766f/amber/ATAC/basset_sample_file.tsv',
                out_peaks='combined_atac.bed', out_filt='dsq_all.bed',
                out_open='dsq_open.bed', out_fin='filt_open_ext.bed',
                out_fa='ext.fa', genome_file='/home/shush/genomes/hg19.fa',
                window=3072, out_dir='.', out_h5='onehot.h5', save_files=True):
    print('Combining IDR beds')
    combine_beds(samplefile, out_peaks)
    print('Filtering in test set chromosomes in the dataset ')
    column_names = filter_dataset(dsq_path, out_filt)
    print('Filtering SNPs in the open chromatin regions')
    bed_intersect(out_filt, out_peaks, out_open)
    print('Extending regions around the SNP')
    dsq_nonneg = extend_ranges(column_names, out_open, out_fin, window)
    print('Converting bed to fa')
    bed_to_fa(out_fin, out_fa, genome_file)
    print('converting fa to one hot encoding')
    coords_list, seqs_list = fasta2list(out_fa)
    onehot_ref, onehot_alt, coord_np = str_to_onehot(coords_list, seqs_list,
                                                    dsq_nonneg, window)
    print('Saving onehots as h5')
    onehot_to_h5(onehot_ref, onehot_alt, coord_np, out_dir, out_h5)

    interm_files = [out_peaks, out_filt, out_open, out_fin, out_fa]
    if save_files:
        for f in interm_files:
            dst_f = os.path.join(out_dir, f)
            shutil.move(f, dst_f)
    else:
        for f in interm_files:
            os.remove(f)
