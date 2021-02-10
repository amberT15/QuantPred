#!/usr/bin/env python

import numpy as np
import pandas as pd
import os
import utils


def make_dir(outdir):
  if not os.path.isdir(outdir):
    os.mkdir(outdir)
  return(outdir)

def get_best(df):
  l_bio_reps = [len(l) for l in df['Biological replicate(s)'].values]
  all_same = all(element == l_bio_reps[0] for element in l_bio_reps)
  if all_same:
    l_tech_reps = [len(l) for l in df['Technical replicate(s)'].values]
    i_best = np.argmax(l_tech_reps)
  else:
    i_best = np.argmax(l_bio_reps)
    return i_best

def process_exp(exp_accession, metadata, assembly, res_dict):
  exp_df = metadata[(metadata['Experiment accession']==exp_accession) & (metadata['File assembly']==assembly)]
  assert exp_df.size > 0, 'Bad accession number, no records found'

  bed = exp_df[exp_df['File format'] == 'bigBed narrowPeak']
  c1 = bed['Output type']=='conservative IDR thresholded peaks'
  c2 = exp_df['Output type'] == 'signal p-value'
  c3 = exp_df['Output type'] == 'fold change over control'
  c4 = exp_df['Output type'] == 'alignments'
  assert any(c1) and any(c2) and any(c3) and any(c4), exp_accession+' has missing data types'
  bed = bed[c1]
  assert bed.shape[0] == 1, 'Multiple conservative IDR thresholded peak bed files identified'
  res_dict['bed'].append(bed)
  sign_bws = exp_df[c2]
  res_dict['sign'].append(sign_bws.iloc[get_best(sign_bws),:])
  fold_bws = exp_df[c3]
  res_dict['fold'].append(fold_bws.iloc[get_best(fold_bws),:])
  # sign_best = get_best(sign_bws)
  bam = exp_df[c4]
  res_dict['bam'].append(bam)

def wget_list(urls, outdir):
  for url in urls:
    h = os.popen('wget -P {} {}'.format(outdir, url))
    h.close()

def save_dataset(res_dict, outdir):
    for prefix, filtered_list in res_dict.items():
        df = pd.concat(filtered_list, axis=1).T
        df.to_csv(os.path.join(outdir, '{}.csv'.format(prefix)))
        prefix_dir = make_dir(os.path.join(outdir, prefix))
        urls = df['File download URL'].values
        wget_list(urls, prefix_dir)

def create_dataset(exp_accession_list, outdir):
  if not os.path.isdir(outdir):
    os.mkdir(outdir)
  assembly = 'GRCh38'
  res_dict = {'sign':[], 'fold':[], 'bam':[], 'bed':[]}
  for exp_accession in exp_accession_list:
    process_exp(exp_accession, metadata, assembly, res_dict)
  save_dataset(res_dict, outdir)



# files_path = './datasets/QQ_encode_TF.txt'
# output_folder = './datasets'
#
# with open(files_path, "r") as file:
#     metadata_url = file.readline()[1:-2] #remove " before and after url
# # file list metadata
# metadata = utils.download_metadata(metadata_url, output_folder)
metadata_path = 'datasets/metadata.tsv'
metadata = pd.read_csv(metadata_path, sep='\t')


exp_accession_list_A549 = ['ENCSR544GUO']
# exp_accession_list_A549 = ['ENCSR544GUO', 'ENCSR000BUB', 'ENCSR035OXA', 'ENCSR886OEO', 'ENCSR593DGU', 'ENCSR192PBJ', 'ENCSR979IOT']
outdir_A549 = 'datasets/A549'
create_dataset(exp_accession_list_A549, outdir_A549)

# exp_accession_list_HepG2 = ['ENCSR544GUO', 'ENCSR000BUB', 'ENCSR035OXA', 'ENCSR886OEO', 'ENCSR593DGU', 'ENCSR192PBJ', 'ENCSR979IOT']
# outdir_HepG2 = 'HepG2'
# create_dataset(exp_accession_list_HepG2, outdir_HepG2)
