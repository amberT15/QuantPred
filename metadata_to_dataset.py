#!/usr/bin/env python
import numpy as np
import pandas as pd
import os
import utils
import sys

def main():
    metadata_path = sys.argv[1]
    output_folder = sys.argv[2]

    # load metadata
    metadata = pd.read_csv(metadata_path, sep=',')
    create_dataset(metadata, output_folder, include=['bed', 'sign', 'fold', 'bam'])

# def make_directory(outdir):
#   if not os.path.isdir(outdir):
#     os.mkdir(outdir)
#   return(outdir)

def make_label(df_row):

    '''
    Add label for each row selected:
    Assay_Experiment target_Biosample term name_Experiment accession
    '''
    label_list = [str(c.values[0]) for c in [df_row['Assay'], df_row['Experiment target'], df_row['Biosample term name'],
                         df_row['Experiment accession']]]
    return('_'.join(label_list).replace(" ", "-"))

def get_same(df, tech_id):
  '''
  Get the same technical replicate as the bam file selected
  '''

  df_i_same = df[df['Technical replicate(s)'] == tech_id]
  return df_i_same

def process_exp(exp_accession, metadata, assembly):
  exp_df = metadata[(metadata['Experiment accession']==exp_accession) & (metadata['File assembly']==assembly)]
  assert exp_df.size > 0, 'Bad accession number, no records found'
  bed = exp_df[(exp_df['File type'] == 'bed') &
                (exp_df['Output type']=='IDR thresholded peaks')]
  sign = exp_df[exp_df['Output type'] == 'signal p-value'] #check if signal bw exists
  fold = exp_df[exp_df['Output type'] == 'fold change over control'] #check if fold bw exists
  bam = exp_df[exp_df['Output type'] == 'alignments'] #check if bam exists
  # throw an error if any one absent
  outputs = [bed, sign, fold, bam]
  assert all([i.shape[0]>0 for i in outputs]), exp_accession+' has missing data types'
  # pick the first bam file
  bam = bam.iloc[[0]]
  id = bam['Technical replicate(s)'].values[0]
  sign = get_same(sign, id)
  fold = get_same(fold, id)
  bed = get_same(bed, id)

  summary_line = [make_label(bed)]
  for output in [bed, sign, fold, bam]:
    summary_line.append(output['File download URL'].values[0])
  return summary_line


def wget_list(urls, outdir):
  for url in urls:
    h = os.popen('wget -P {} {}'.format(outdir, url))
    h.close()

def get_filepaths(urls, filedir):
    filepaths = []
    for url in urls:
        print(url.split('/')[-1])
        filepaths.append(os.path.abspath(os.path.join(filedir, url.split('/')[-1])))
    return filepaths


def save_dataset(res_dict, outdir):
  for prefix, filtered_list in res_dict.items():
    print("Prcessing set labelled {}".format(prefix))
    df = pd.concat(filtered_list, axis=1)
    prefix_dir = utils.make_directory(os.path.join(outdir, prefix))
    df.to_csv(os.path.join(prefix_dir, '{}.csv'.format(prefix)))

    urls = df['File download URL'].values

    wget_list(urls, prefix_dir)

def create_dataset(metadata, outdir, folder_label='summary',
                    include=['fold', 'sign'], assembly = 'GRCh38'):
  # include all files for generatign the summary file
  cols = ['label','bed', 'sign', 'fold', 'bam']
  # init summary list to save the experiment ID - file relationship
  summary = []
  # create output folder if not present
  utils.make_directory(outdir)
  # loop over the list of experiments
  exp_accession_list = list(set(metadata['Experiment accession'].values))

  for exp_accession in exp_accession_list:
    # filter files and save selection
    summary.append(process_exp(exp_accession, metadata, assembly))

  sum_df = pd.DataFrame(summary, columns=cols)
  sum_df.to_csv(os.path.join(outdir, folder_label+'.csv'))
  cols = include

  for i in range(len(cols)):

      data_subdir = utils.make_directory(os.path.join(outdir, cols[i])) # create file type dir
      wget_list(sum_df[cols[i]].values, data_subdir) # download URLs
  filepaths = get_filepaths(sum_df['bed'].values, data_subdir)
  with open(os.path.join(outdir, 'basset_sample_beds.txt'), 'w') as filehandle:
      for i in range(len(filepaths)):
          filehandle.write('{}\t{}\n'.format(sum_df['label'][i], filepaths[i]))

if __name__=='__main__':
    main()
