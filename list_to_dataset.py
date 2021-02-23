#!/usr/bin/env python
import numpy as np
import pandas as pd
import os
import utils


def main():
    files_path = 'datasets/QQ_encode_TF.txt'
    output_folder = 'datasets'

    with open(files_path, "r") as file:
        metadata_url = file.readline()[1:-2] #remove " before and after url
    # file list metadata
    metadata = utils.download_metadata(metadata_url, output_folder)
    # metadata = pd.read_csv(metadata_path, sep='\t')


    # exp_accession_list_A549 = ['ENCSR544GUO', 'ENCSR000BUB']
    # exp_accession_list_A549 = ['ENCSR544GUO', 'ENCSR000BUB', 'ENCSR035OXA', 'ENCSR886OEO', 'ENCSR593DGU', 'ENCSR192PBJ', 'ENCSR979IOT']
    # outdir_A549 = 'datasets/A549'
    # create_dataset(exp_accession_list_A549, outdir_A549)

    exp_accession_list_HepG2 = ['ENCSR544GUO', 'ENCSR000BUB', 'ENCSR035OXA', 'ENCSR886OEO', 'ENCSR593DGU', 'ENCSR192PBJ', 'ENCSR979IOT']
    outdir_HepG2 = 'datasets/HepG2'
    create_dataset(exp_accession_list_HepG2, outdir_HepG2, metadata, include=['bam'])

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
  exp_df.to_csv('exp.csv')
  assert exp_df.size > 0, 'Bad accession number, no records found'
  bed = exp_df[(exp_df['File type'] == 'bed') &
                (exp_df['Output type']=='IDR thresholded peaks')]
  sign = exp_df[exp_df['Output type'] == 'signal p-value'] #check if signal bw exists
  fold = exp_df[exp_df['Output type'] == 'fold change over control'] #check if fold bw exists
  fold.to_csv('fold0.csv')
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

def create_dataset(exp_accession_list, outdir, metadata, folder_label='summary',
                    include=['fold', 'sign'], assembly = 'GRCh38'):
  # include all files for generatign the summary file
  cols = ['label','bed', 'sign', 'fold', 'bam']
  # init summary list to save the experiment ID - file relationship
  summary = []
  # create output folder if not present
  utils.make_directory(outdir)
  # loop over the list of experiments defined
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
