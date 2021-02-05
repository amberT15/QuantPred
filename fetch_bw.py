#!/usr/bin/env python
from optparse import OptionParser
import os
import pandas as pd
import sys
import subprocess
from pathlib import Path

output_path = sys.argv[1]
print(output_path)
exp = output_path.split('/')[-1]
print("Processing folder {}".format(exp))

assert os.path.isdir(output_path), "Experiment folder not found"
metadata_path = output_path + '/metadata.tsv'
filtered_path = output_path + '/{}_filtered_df.csv'.format(exp)
bw_output_dir = os.path.join(output_path, 'bigwigs')
print(bw_output_dir)

filtered_df = pd.read_csv(filtered_path)
metadata_df = pd.read_csv(metadata_path, sep='\t')

bw_list = []
for i, row in filtered_df.iterrows():
    exp_name = row['Experiment accession']
    exp_df = metadata_df[metadata_df['Experiment accession']==exp_name]
    exp_df = exp_df[(exp_df['File format']=='bigWig') &
                  (exp_df['File assembly']=='hg19') &
                  (exp_df['Output type']=='fold change over control')]

    bw_list.append(exp_df[exp_df['Technical replicate(s)'].str.contains(",")])

bw_df = pd.concat(bw_list)
bw_df.to_csv(os.path.join(output_path, 'bw_filtered.csv'))

if not os.path.isdir(bw_output_dir):
    os.mkdir(bw_output_dir)
for url in bw_df['File download URL']:
    one_output_path = os.path.join(bw_output_dir, url.split('/')[-1])
    # download url to the output dir
    cmd = 'wget -O {} {}'.format(one_output_path, url)
    subprocess.call(cmd, shell=True)
