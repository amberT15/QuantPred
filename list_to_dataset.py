#!/usr/bin/env python

import numpy as np
import pandas as pd
import os
import utils


def make_dir(outdir):
  if not os.path.isdir(outdir):
    os.mkdir(outdir)
  return(outdir)

def make_label(df_row):

    '''
    Add label for each row selected:
    Assay_
    Experiment target_
    Biosample term name_
    Experiment accession_
    '''
    label_list = [str(c.values[0]) for c in [df_row['Assay'], df_row['Experiment target'], df_row['Biosample term name'],
                         df_row['Experiment accession']]]
    return('_'.join(label_list).replace(" ", "-"))

def get_best(df):
  l_bio_reps = [len(l) for l in df['Biological replicate(s)'].values]
  all_same = all(element == l_bio_reps[0] for element in l_bio_reps)
  if all_same:
    l_tech_reps = [len(l) for l in df['Technical replicate(s)'].values]
    i_best = np.argmax(l_tech_reps)
  else:
    i_best = np.argmax(l_bio_reps)
    return i_best

def process_exp(exp_accession, metadata, assembly):
  exp_df = metadata[(metadata['Experiment accession']==exp_accession) & (metadata['File assembly']==assembly)]
  assert exp_df.size > 0, 'Bad accession number, no records found'

  bed = exp_df[(exp_df['File format'] == 'bigBed narrowPeak') &
                (exp_df['Output type']=='conservative IDR thresholded peaks')]
  sign = exp_df[exp_df['Output type'] == 'signal p-value'] #check if signal bw exists
  fold = exp_df[exp_df['Output type'] == 'fold change over control'] #check if fold bw exists
  bam = exp_df[exp_df['Output type'] == 'alignments'] #check if bam exists
  # throw an error if any one absent
  outputs = [bed, sign, fold, bam]
  assert all([i.shape[0]>0 for i in outputs]), exp_accession+' has missing data types'
  # check if multiple exist, which there shouldn't be?
  assert bed.shape[0] == 1, 'Multiple conservative IDR thresholded peak bed files identified'
  sign = sign.iloc[[get_best(sign)]]
  fold = fold.iloc[[get_best(fold)]]
  bam = bam.iloc[[0]]
  summary_line = [make_label(bed)]
  summary_line = summary_line + [i['File download URL'].values[0] for i in outputs]
  return summary_line



def wget_list(urls, outdir):
  for url in urls:
    h = os.popen('wget -P {} {}'.format(outdir, url))
    h.close()

def save_dataset(res_dict, outdir):
  for prefix, filtered_list in res_dict.items():
    print("Prcessing set labelled {}".format(prefix))
    df = pd.concat(filtered_list, axis=1)
    prefix_dir = make_dir(os.path.join(outdir, prefix))
    df.to_csv(os.path.join(prefix_dir, '{}.csv'.format(prefix)))

    urls = df['File download URL'].values

    wget_list(urls, prefix_dir)

def create_dataset(exp_accession_list, outdir, folder_label='summary'):
  cols = ['label','bed', 'sign', 'fold', 'bam']
  summary = []
  if not os.path.isdir(outdir):
    os.mkdir(outdir)
  assembly = 'GRCh38'
  for exp_accession in exp_accession_list:
    summary.append(process_exp(exp_accession, metadata, assembly))
  sum_df = pd.DataFrame(summary, columns=cols)
  sum_df.to_csv(os.path.join(outdir, folder_label+'.csv'))
  for i in range(1, len(cols)):
      data_subdir = make_dir(os.path.join(outdir, cols[i]))
      wget_list(sum_df[cols[i]].values, data_subdir)



files_path = 'datasets/QQ_encode_TF.txt'
output_folder = 'datasets'

with open(files_path, "r") as file:
    metadata_url = file.readline()[1:-2] #remove " before and after url
# file list metadata
metadata = utils.download_metadata(metadata_url, output_folder)
metadata = pd.read_csv(metadata_path, sep='\t')


# exp_accession_list_A549 = ['ENCSR544GUO', 'ENCSR000BUB']
exp_accession_list_A549 = ['ENCSR544GUO', 'ENCSR000BUB', 'ENCSR035OXA', 'ENCSR886OEO', 'ENCSR593DGU', 'ENCSR192PBJ', 'ENCSR979IOT']
outdir_A549 = 'datasets/A549'
create_dataset(exp_accession_list_A549, outdir_A549)

exp_accession_list_HepG2 = ['ENCSR544GUO', 'ENCSR000BUB', 'ENCSR035OXA', 'ENCSR886OEO', 'ENCSR593DGU', 'ENCSR192PBJ', 'ENCSR979IOT']
outdir_HepG2 = 'datasets/HepG2'
create_dataset(exp_accession_list_HepG2, outdir_HepG2)
