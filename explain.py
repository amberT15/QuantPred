import numpy as np
import tensorflow as tf
from tensorflow import keras
import math
import matplotlib.pyplot as plt
import pandas as pd
import logomaker
import subprocess
import os, shutil, h5py,scipy
import util
import custom_fit
import seaborn as sns

def task_robustness(selected_read,model, task_idx, batch_size = 50, shift_num = 10, window_size = 2048, visualize = True, smooth_saliency = True):
    var_saliency_list = []
    var_pred_list = []
    chop_size = selected_read.shape[1]
    center_idx = int(0.5*(chop_size-window_size))
    center_range = np.array(range(center_idx,center_idx+window_size))
    conserve_size = window_size*2 - chop_size
    conserve_start = chop_size//2 - conserve_size//2
    conserve_end = conserve_start + conserve_size-1

    i = 0
    while i < len(selected_read):
        if i+ batch_size < len(selected_read):
            seq = selected_read[i:i+batch_size]
            #target = selected_target[i:i+batch_size][:,:,task_idx]
            batch_n = batch_size
            i = i+batch_size
        else:
            seq = selected_read[i:len(selected_read)]
            #target = selected_target[i:len(selected_read)][:,:,task_idx]
            batch_n = len(selected_read) - i
            i = len(selected_read)


        shifted_seq,_,shift_idx = util.window_shift(seq,seq,window_size,shift_num)
        #get prediction for shifted read
        shift_pred = model.predict(shifted_seq)[:,:,task_idx]
        bin_size = window_size / shift_pred.shape[1]
        shift_pred = np.repeat(shift_pred,bin_size,axis = 1)

        #get saliency for shifted read
        shift_saliency = complete_saliency(shifted_seq,model,class_index = task_idx)
        shift_saliency = shift_saliency * shifted_seq

        #Select conserve part only
        crop_start_i = conserve_start - shift_idx - center_idx
        crop_idx = crop_start_i[:,None] + np.arange(conserve_size)
        crop_idx = crop_idx.reshape(conserve_size*shift_num*batch_n)
        crop_row_idx = np.repeat(range(0,shift_num*batch_n),conserve_size)
        crop_f_index = np.vstack((crop_row_idx,crop_idx)).T.reshape(shift_num*batch_n,conserve_size,2)

        #get saliency 1k part
        shift_saliency_1k=tf.gather_nd(shift_saliency,crop_f_index)

        sep_saliency =np.array(np.array_split(shift_saliency_1k,batch_n))
        average_saliency = np.average(np.array(sep_saliency),axis = 1)

        var_saliency = np.std(np.sum(sep_saliency,axis = -1),axis = 1)
        var_saliency_sum = np.sum(var_saliency,axis = 1)

        #get pred 1k part
        shift_pred_1k=tf.gather_nd(shift_pred,crop_f_index)
        sep_pred = np.array(np.array_split(shift_pred_1k,batch_n))
        var_pred = np.std(sep_pred,axis = 1)
        var_pred_sum = np.sum(var_pred,axis = 1)

        #add var result to list
        var_saliency_list.append(var_saliency_sum)
        var_pred_list.append(var_pred_sum)

        if visualize == True:
        #make 2 subplots per sequence
            for a in range(0,batch_n):

                fig, (ax1, ax2,ax3) = plt.subplots(3,1,figsize = (15,6))
                # if ground_truth == True:
                #     #plot ground truth pred
                #     sns.lineplot(x = range(0,chop_size),
                #                 y = np.squeeze(target[a]),ax = ax1,color = 'lightblue')


                for shift_n in range(0,shift_num):
                    #visualize prediction
                    sns.lineplot(x = center_range + shift_idx[shift_n],
                                 y = shift_pred[a*shift_num + shift_n,:],ax = ax1,
                                 alpha = 0.35)
                    #visualize saliency
                    tmp_saliency = shift_saliency[a*shift_num + shift_n]
                    sns.lineplot(x = center_range + shift_idx[shift_n],
                                y =np.sum(tmp_saliency.numpy(),axis = 1),ax=ax2,
                                alpha = 0.35)

                if smooth_saliency==True:
                    #plot average saliency
                    sns.lineplot(x = range(conserve_start,conserve_end+1),
                             y = np.sum(average_saliency[a],axis = 1),
                             ax = ax2, color = 'lightblue' )

                line_saliency = np.sum(average_saliency[a],axis = 1)

                sns.lineplot(x = range(0,conserve_size),
                             y = line_saliency,
                             ax = ax3)
                ax3.fill_between(range(0,conserve_size),
                                line_saliency-var_saliency[a],
                                line_saliency+var_saliency[a], alpha=.8,
                                color = 'black')

                plt.tight_layout()
                plt.show()

    return np.hstack(var_saliency_list),np.hstack(var_pred_list)

def batch_robustness_test(selected_read,selected_target,model,visualize = True,ground_truth = True, batch_size = 50,
                            smooth_saliency = True, shift_num = 10, window_size = 2048):
    var_saliency_list = []
    var_pred_list = []
    chop_size = selected_read.shape[1]
    center_idx = int(0.5*(chop_size-window_size))
    center_range = np.array(range(center_idx,center_idx+window_size))
    conserve_size = window_size*2 - chop_size
    conserve_start = chop_size//2 - conserve_size//2
    conserve_end = conserve_start + conserve_size-1

    i = 0
    while i < len(selected_read):
        if i+ batch_size < len(selected_read):
            seq = selected_read[i:i+batch_size]
            target = selected_target[i:i+batch_size]
            batch_n = batch_size
            i = i+batch_size
        else:
            seq = selected_read[i:len(selected_read)]
            target = selected_target[i:len(selected_read)]
            batch_n = len(selected_read) - i
            i = len(selected_read)


        shifted_seq,_,shift_idx = util.window_shift(seq,seq,window_size,shift_num)
        #get prediction for shifted read
        shift_pred = model.predict(shifted_seq)
        bin_size = window_size / shift_pred.shape[1]
        shift_pred = np.repeat(shift_pred,bin_size,axis = 1)

        #get saliency for shifted read
        center_seq,_ = custom_fit.center_crop(seq,seq,window_size)
        center_pred = model.predict(center_seq)
        short_max_task = np.argmax(np.sum(center_pred,axis=1),axis = 1)
        max_task = np.repeat(short_max_task,shift_num)
        shift_saliency = complete_saliency(shifted_seq,model,class_index = max_task[0])
        shift_saliency = shift_saliency * shifted_seq

        #Select conserve part only
        crop_start_i = conserve_start - shift_idx - center_idx
        crop_idx = crop_start_i[:,None] + np.arange(conserve_size)
        crop_idx = crop_idx.reshape(conserve_size*shift_num*batch_n)
        crop_row_idx = np.repeat(range(0,shift_num*batch_n),conserve_size)
        crop_f_index = np.vstack((crop_row_idx,crop_idx)).T.reshape(shift_num*batch_n,conserve_size,2)

        #get saliency 1k part
        shift_saliency_1k=tf.gather_nd(shift_saliency,crop_f_index)

        sep_saliency =np.array(np.array_split(shift_saliency_1k,batch_n))
        average_saliency = np.average(np.array(sep_saliency),axis = 1)

        var_saliency = np.var(np.sum(sep_saliency,axis = -1),axis = 1)
        var_saliency_sum = np.sum(var_saliency,axis = 1)

        #get pred 1k part
        shift_pred_1k=tf.gather_nd(shift_pred[range(shift_pred.shape[0]),:,max_task],crop_f_index)
        sep_pred = np.array(np.array_split(shift_pred_1k,batch_n))
        var_pred = np.var(sep_pred,axis = 1)
        var_pred_sum = np.sum(var_pred,axis = 1)

        #add var result to list
        var_saliency_list.append(var_saliency_sum)
        var_pred_list.append(var_pred_sum)

        if visualize == True:
        #make 2 subplots per sequence
            for a in range(0,batch_n):

                fig, (ax1, ax2,ax3) = plt.subplots(3,1,figsize = (15,15))
                if ground_truth == True:
                    #plot ground truth pred
                    # sns.lineplot(x = range(500,1024),
                    #             y = np.squeeze(target[a,:,short_max_task[a]])[1524:2048],ax = ax1,color = 'black')
                    sns.lineplot(x = range(0,1024),y = np.squeeze(target[a,:,short_max_task[a]]),ax = ax1,color = 'black')
                    ax1.set(xlabel='Position', ylabel='Coverage')


                for shift_n in range(0,shift_num):
                    #visualize prediction
                    sns.lineplot(x = center_range + shift_idx[a*shift_num+shift_n],
                                 y = shift_pred[a*shift_num + shift_n,:,short_max_task[a]],ax = ax1,
                                 alpha = 0.35)
                    # shift_i = shift_idx[a*shift_num+shift_n]
                    # sns.lineplot(x = range(500,1024),
                    #             y = shift_pred[a*shift_num + shift_n,:,short_max_task[a]][512-shift_i+500:1536-shift_i],
                    #             ax = ax1,alpha = 0.35)

                    #visualize saliency
                    tmp_saliency = shift_saliency[a*shift_num + shift_n]
                    sns.lineplot(x = center_range + shift_idx[shift_n],
                                y =np.sum(tmp_saliency.numpy(),axis = 1),ax=ax2,
                                alpha = 0.35)

                if smooth_saliency==True:
                    #plot average saliency
                    sns.lineplot(x = range(conserve_start,conserve_end+1),
                             y = np.sum(average_saliency[a],axis = 1),
                             ax = ax2, color = 'lightblue' )

                line_saliency = np.sum(average_saliency[a],axis = 1)

                sns.lineplot(x = range(0,conserve_size),
                             y = line_saliency,
                             ax = ax3)
                ax3.fill_between(range(0,conserve_size),
                                line_saliency-var_saliency[a],
                                line_saliency+var_saliency[a], alpha=.8,
                                color = 'black')

                plt.tight_layout()
                plt.show()

    return np.concatenate(var_saliency_list), np.concatenate(var_pred_list)

def plot_saliency(saliency_map):

    fig, axs = plt.subplots(saliency_map.shape[0],1,figsize=(200,5*saliency_map.shape[0]))
    for n, w in enumerate(saliency_map):
        if saliency_map.shape[0] == 1:
            ax = axs
        else:
            ax = axs[n]
        #plot saliency map representation
        saliency_df = pd.DataFrame(w, columns = ['A','C','G','T'])
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

def vcf_robust(ref,alt,model,shift_num=10,window_size=2048):
    #calculate the coordinates for sequences to conserve in the center
    vcf_diff_list = []
    chop_size = ref.shape[1]
    center_idx = int(0.5*(chop_size-window_size))
    center_range = np.array(range(center_idx,center_idx+window_size))
    conserve_size = window_size*2 - chop_size
    conserve_start = chop_size//2 - conserve_size//2
    conserve_end = conserve_start + conserve_size-1

    for i, (ref_seq, alt_seq) in enumerate(zip(ref,alt)):
        #creat shifted sequence list and make predictions
        shifted_ref,shifted_alt,shift_idx = util.window_shift(ref_seq,alt_seq,window_size,shift_num,both_seq = True)
        ref_pred = model.predict(shifted_ref)
        alt_pred = model.predict(shifted_alt)
        bin_size = window_size / ref_pred.shape[1]
        ref_pred = np.repeat(ref_pred,bin_size,axis = 1)
        alt_pred = np.repeat(alt_pred,bin_size,axis = 1)

        # #get the maximum cell line that VCF analysis will be done on
        # center_seq,_ = custom_fit.center_crop(ref_seq,alt_seq,window_size)
        # center_pred = model.predict(center_seq)
        # max_task = np.argmax(np.sum(center_pred,axis=1),axis = 1)

        #Select conserved part
        crop_start_i = conserve_start - shift_idx - center_idx
        crop_idx = crop_start_i[:,None] + np.arange(conserve_size)
        crop_idx = crop_idx.reshape(conserve_size*shift_num)
        crop_row_idx = np.repeat(range(0,shift_num),conserve_size)
        crop_f_index = np.vstack((crop_row_idx,crop_idx)).T.reshape(shift_num,conserve_size,2)

        #get pred 1k part
        ref_pred_1k=tf.gather_nd(ref_pred,crop_f_index)
        alt_pred_1k=tf.gather_nd(alt_pred,crop_f_index)

        #get average pred
        avg_ref = np.mean(ref_pred_1k,axis=0)
        avg_alt = np.mean(alt_pred_1k,axis=0)

        #get difference between average coverage value
        vcf_diff = np.sum(avg_alt,axis = 0) - np.sum(avg_ref,axis = 0)
        vcf_diff_list.append(vcf_diff)

    return vcf_diff_list

def vcf_test(ref,alt,coords,model,background_size = 100):
    # score for reference and alternative allele
    ref_pred = model.predict(ref)
    alt_pred = model.predict(alt)

    ref_pred_cov = np.sum(ref_pred,axis = 1)
    alt_pred_cov = np.sum(alt_pred,axis = 1)
    max_task = np.argmax(ref_pred_cov,axis = 1)
    ref_max_cov = ref_pred_cov[range(0,ref.shape[0]),max_task]
    alt_max_cov = alt_pred_cov[range(0,ref.shape[0]),max_task]

    d = {'chromosome': coords[:,0], 'start': coords[:,1],'end':coords[:,2]}
    df = pd.DataFrame(data=d)

    df['ref'] = ref_max_cov
    df['alt'] = alt_max_cov

    #mutate very edge regions
    background_distribution = []
    for i,ref_seq in enumerate(ref):
        mut_loci = np.random.randint(500,1023,size = background_size)
        direction = np.random.choice([-1,1],size = background_size)
        mut_loci = len(ref_seq)/2 + mut_loci * direction
        mut_loci = mut_loci.astype('int')
        mut_batch = np.tile(ref_seq,(background_size,1,1))
        mut_row = mut_batch[range(0,background_size),mut_loci]
        ori_empty_base = np.where(mut_row!= 1)[1].reshape(mut_row.shape[0],3)
        mut_base = np.apply_along_axis(np.random.choice, axis=1, arr=ori_empty_base, size=1)
        mut_batch[range(0,background_size),mut_loci] = [0,0,0,0]
        mut_batch[range(0,background_size),mut_loci,mut_base] = 1

        mut_pred = model.predict(mut_batch)
        mut_pred_cov = np.sum(mut_pred,axis =1)[:,max_task[i]]
        background_distribution.append(mut_pred_cov)

    df['background'] = background_distribution
    return df




def visualize_vcf(ref,alt,model,background_size = 100,title = None):
    #ref and alternative prediction for the task with most signal
    ref = tf.expand_dims(ref,axis=0)
    alt = tf.expand_dims(alt,axis=0)
    ref_pred = model.predict(ref)
    alt_pred = model.predict(alt)
    ref_pred_cov = np.sum(ref_pred,axis = 1)
    alt_pred_cov = np.sum(alt_pred,axis = 1)
    max_task = np.argmax(ref_pred_cov,axis = 1)
    ref_pred = np.squeeze(ref_pred[:,:,max_task])
    alt_pred = np.squeeze(alt_pred[:,:,max_task])

    #generate random background mutation
    background_distribution = []
    for i,ref_seq in enumerate(ref):
        mut_loci = np.random.randint(500,1023,size = background_size)
        direction = np.random.choice([-1,1],size = background_size)
        mut_loci = len(ref_seq)/2 + mut_loci * direction
        mut_loci = mut_loci.astype('int')
        mut_batch = np.tile(ref_seq,(background_size,1,1))
        mut_row = mut_batch[range(0,background_size),mut_loci]
        ori_empty_base = np.where(mut_row!= 1)[1].reshape(mut_row.shape[0],3)
        mut_base = np.apply_along_axis(np.random.choice, axis=1, arr=ori_empty_base, size=1)
        mut_batch[range(0,background_size),mut_loci] = [0,0,0,0]
        mut_batch[range(0,background_size),mut_loci,mut_base] = 1

        mut_pred = model.predict(mut_batch)[:,:,max_task]

    fig = plt.fill_between(range(0,mut_pred.shape[1]),
                    np.squeeze(mut_pred.max(axis = 0)),
                    np.squeeze(mut_pred.min(axis = 0)),facecolor='grey')

    plt.plot(ref_pred,label = 'reference', color = 'black')
    plt.plot(alt_pred,label = 'alternative',color = 'red')
    plt.xlabel('Position')
    plt.ylabel('Coverage')
    plt.legend()
    if title:
        plt.title(title)

def complete_saliency(X,model,class_index,func = tf.math.reduce_mean):
  """fast function to generate saliency maps"""
  # if not tf.is_tensor(X):
  #   X = tf.Variable(X)

  X = tf.cast(X, dtype='float32')

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

# def vcf_pct(vcf_df):
#     pct_list = []
#     for (i,alt) in enumerate(vcf_df['alt']):
#         small_pct = len(np.where(np.array(vcf_df['background'][i]) < alt)[0])
#         large_pct = len(np.where(np.array(vcf_df['background'][i]) > alt)[0])
#         pct_list.append(np.minimum(small_pct,large_pct)/100)
#     vcf_df['pct'] = pct_list
#     return vcf_df
#
# def merge_background_vcf(vcf_df):
#     centered_background = vcf_df['background'] - vcf_df['ref']
#     centered_background = np.concatenate(centered_background)
#     centered_alt = np.array(vcf_df['alt'] - vcf_df['ref'])
#
#     pct_merged = []
#     for alt_i in centered_alt:
#         small_pct = len(np.where(centered_background < alt_i)[0])
#         large_pct = len(np.where(centered_background > alt_i)[0])
#         pct_merged.append(np.minimum(small_pct,large_pct)/len(centered_background))
#
#     vcf_df['pct_merged'] = pct_merged
#     return vcf_df


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
    combined_csv = pd.concat([(pd.read_csv(f, sep='\t', header=None).iloc[:,:3]).drop_duplicates() for f in bed_paths ])
    combined_csv.to_csv(out_path, sep='\t', header=None, index=None)

def filter_dataset(dsq_path, out_path='dsq_all.bed'):
    dsq_all = pd.read_csv(dsq_path, sep='\t')
    dsq_all['ID'] = dsq_all.index
    dsq_filt = dsq_all[(dsq_all['chrom']=='chr8')]
    dsq_filt[['a1','a2']] = dsq_filt['genotypes'].str.split('/',expand=True) # write into separate columns
    dsq_filt.to_csv(out_path, sep='\t', header=False, index=None)
    return dsq_filt

def bed_intersect(dataset_bed, comb_peak, out_path):
    bashCmd = "bedtools intersect -wa -a {} -b {} > {}".format(dataset_bed, comb_peak, out_path)
    process = subprocess.Popen(bashCmd, shell=True)
    output, error = process.communicate()
    print(error)

def extend_ranges(column_names, bedfile, out_path, window):
    dsq_df = pd.read_csv(bedfile, sep='\t', header=None, index_col=None)
    dsq_df.columns = column_names #list(dsq_filt)
    dsq_filt = dsq_df[['chrom', 'snpChromStart', 'snpChromEnd', 'a1', 'a2',
                        'strand', 'rsid', 'pred.fit.pctSig','ID']]
    # get the 3K range start and end
    start, end = enforce_const_range(dsq_filt['snpChromEnd']-1, window)
    dsq_ext = dsq_filt.copy()
    #
    dsq_ext.iloc[:,1] = start.values
    dsq_ext.iloc[:,2] = end.values
    dsq_nonneg = dsq_ext[dsq_ext['snpChromStart']>0]
    dsq_nonneg = dsq_nonneg.reset_index(drop=True)
    dsq_nonneg['counts'] = dsq_nonneg.groupby(['chrom', 'snpChromStart'])['snpChromStart'].transform('count').values
    dsq_nonneg = dsq_nonneg.drop_duplicates().reset_index(drop=True)
    dsq_nonneg.to_csv(out_path, header=None, sep='\t', index=None)
    counts_per_cell = dsq_nonneg['counts'].values
    pct_sign = dsq_nonneg['pred.fit.pctSig'].values
    return dsq_nonneg, counts_per_cell, pct_sign

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


def onehot_to_h5(onehot_ref, onehot_alt, coord_np, pct_sign, out_dir='.', filename='onehot.h5'):
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    onehot_ref_alt = h5py.File(os.path.join(out_dir, filename), 'w')
    onehot_ref_alt.create_dataset('ref', data=onehot_ref, dtype='float32')
    onehot_ref_alt.create_dataset('alt', data=onehot_alt, dtype='float32')
    onehot_ref_alt.create_dataset('fasta_coords', data=coord_np, dtype='i')
    # onehot_ref_alt.create_dataset('cell_lines', data=cell_lines, dtype='i')
    onehot_ref_alt.create_dataset('pct_sign', data=pct_sign, dtype='float32')
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
    dsq_nonneg, counts_per_cell, pct_sign = extend_ranges(column_names, out_open, out_fin, window)
    print('Converting bed to fa')
    bed_to_fa(out_fin, out_fa, genome_file)
    print('converting fa to one hot encoding')
    coords_list, seqs_list = fasta2list(out_fa)
    onehot_ref, onehot_alt, coord_np = str_to_onehot(coords_list, seqs_list,
                                                    dsq_nonneg, window)
    print('Saving onehots as h5')
    onehot_to_h5(onehot_ref, onehot_alt, coord_np, counts_per_cell, pct_sign, out_dir, out_h5)

    interm_files = [out_peaks, out_filt, out_open, out_fin, out_fa]
    if save_files:
        for f in interm_files:
            dst_f = os.path.join(out_dir, f)
            shutil.move(f, dst_f)
    else:
        for f in interm_files:
            os.remove(f)

def merge_one_cell_line(chr8_dsq_file, idr_bed, i='0', out_merged_bed='merged.bed'):
    assert isinstance(i, str), 'Input str as identifier!'

    bashCmd = "bedtools intersect -wa -a {} -b {} > {}".format(chr8_dsq_file, idr_bed, out_merged_bed)
    process = subprocess.Popen(bashCmd, shell=True)
    output, error = process.communicate()
    chr8_dsq = pd.read_csv(chr8_dsq_file,  header=None, sep='\t')
    # keep_cols = ['chrom', 'snpChromStart', 'snpChromEnd']
    keep_cols = ['chrom', 'snpChromStart', 'snpChromEnd', 'rsid',
                 'pred.fit.pctSig', 'strand', 'motifname',
                 'position', 'genotypes', 'ID', 'a1', 'a2']
    chr8_dsq.columns = keep_cols
    merged = pd.read_csv('merged.bed', header=None, sep='\t')
    merged.columns = keep_cols
    merged['idr_N'] = i
    merged_dsq = chr8_dsq.merge(merged, how='outer')
    open_vcfs = merged_dsq[merged_dsq['idr_N']==i]
    open_vcfs = open_vcfs.drop_duplicates().reset_index(drop=True)
    return open_vcfs

def get_h5_with_cells(dsq_path, window=3072, out_dir='.',
                      samplefile='/mnt/906427d6-fddf-41bf-9ec6-c3d0c37e766f/amber/ATAC/basset_sample_file.tsv',
                      save_files=False):
    dsq = pd.read_csv(dsq_path, sep='\t') # dataframe from the paper
    dsq_chr8 = filter_dataset(dsq_path, 'dsq_chr8.bed') # filter chr8 VCFs
    # list to save per cell line vcfs in the open chromatin regions
    per_cell_open_vcfs = []
    # open samplefile and get IDR filepaths
    bed_paths = pd.read_csv(samplefile, sep='\t', header=None)[1].values
    # per cell line
    for f, bed_file in enumerate(bed_paths):
        # get VCFs in the open chromatin region
        open_vcfs = merge_one_cell_line('dsq_chr8.bed', bed_file, i=str(f))
        per_cell_open_vcfs.append(open_vcfs) #save it
    # put all into one df
    conc_vcfs = pd.concat(per_cell_open_vcfs)
    # merge redundant ones and collect cell line info
    merged_vcfs = conc_vcfs.groupby(['chrom','snpChromStart','snpChromEnd'])['idr_N'].apply(', '.join).reset_index()
    # reattach columsn with metadata
    complete_df = merged_vcfs.merge(conc_vcfs, how='left', on=['chrom', 'snpChromStart', 'snpChromEnd'])
    # remove redundant columns and rows
    complete_unq = complete_df.drop(columns='idr_N_y').drop_duplicates().reset_index(drop=True)
    # get the 3K range start and end
    start, end = enforce_const_range(complete_unq['snpChromEnd']-1, window)
    complete_unq.insert(1, '3K_start', start.values) # add starts
    complete_unq.insert(2, '3K_end', end.values) # add ends
    complete_unq = complete_unq[complete_unq['3K_start']>0] # remove ones starting at negative coords
    complete_unq.rename(columns={'idr_N_x':'cell_lines'}, inplace=True) # rename column
    complete_unq.to_csv(os.path.join(out_dir, 'vcf_metadata.csv'), sep='\t', index=None) # save the complete metadata table
    # save version needed for fa conversion
    complete_unq[['chrom', '3K_start', '3K_end', 'rsid','pred.fit.pctSig',
                 'strand']].to_csv('out_fin.bed', sep='\t', header=None, index=None)
    bed_to_fa('out_fin.bed', 'out.fa', genome_file='/home/shush/genomes/hg19.fa')
    coords_list, seqs_list = fasta2list('out.fa')
    onehot_ref, onehot_alt, coord_np = str_to_onehot(coords_list, seqs_list,
                                                    complete_unq, window)

    onehot_to_h5(onehot_ref, onehot_alt, coord_np,
                np.array(complete_unq['pred.fit.pctSig'].values), out_dir)
    shutil.copy(dsq_path, os.path.join(out_dir, dsq_path))
    interm_files = ['dsq_chr8.bed', 'out_fin.bed', 'out.fa']
    if save_files:
        for f in interm_files:
            dst_f = os.path.join(out_dir, f)
            shutil.move(f, dst_f)
    else:
        for f in interm_files:
            os.remove(f)
