#!/usr/bin/env python
from optparse import OptionParser
import os
import urllib.request
import pandas as pd
import sys
import utils

def main():
    usage = 'usage: %prog [options] <files_path> <output_folder>'
    parser = OptionParser(usage)
    parser.add_option('-g', dest='gen_assembly',
        default='hg19', type='str',
        help='Genome assembly [Default: %default]')
    parser.add_option('-b', dest='biosample',
        default='', type='str',
        help='Biosample to search and keep [Default: %default]')
    parser.add_option('--no_crispr', action="store_true", dest='crispr',
        help='Filter Biosample genetic modifications methods [Default: %default]')
    parser.add_option('-l', dest='limit',
        default=None, type='int',
        help='Limit the size of a dataset [Default: %default]')
    parser.add_option('-s', dest='seed',
        default=42, type='int',
        help='Seed the random choice of dataset rows [Default: %default]')
    (options, args) = parser.parse_args()
    if len(args) != 2:
        parser.error('Must provide file list from ENCODE and output folder')
    else:
        files_path = args[0]
        output_folder = args[1]

    utils.make_directory(output_folder)
    base_dir = utils.get_parent(files_path)
    # file from encode download
    with open(files_path, "r") as file:
        metadata_url = file.readline()[1:-2] #remove " before and after url
    # file list metadata
    metadata = download_metadata(metadata_url, output_folder)
    metadata['Experiment accession and replicate'] = metadata['Experiment accession']+'_'+ metadata['Biological replicate(s)']

    assay_groups = metadata.groupby(by='Assay')
    #TODO: move this into a json file
    # crit_dict = {'DNase-seq':['peaks'],
    #              'TF ChIP-seq':['conservative IDR thresholded peaks',
    #                             'optimal IDR thresholded peaks',
    #                             'pseudoreplicated IDR thresholded peaks'],
    #              'Histone ChIP-seq':['replicated peaks',
    #                                  'pseudo-replicated peaks']}

    # Only highest quality assays
    crit_dict = {'DNase-seq':['peaks'],
                 'TF ChIP-seq':['conservative IDR thresholded peaks'],
                 'Histone ChIP-seq':['replicated peaks']}
    df_filt = []
    for assay, assay_df in assay_groups:
        # TODO:ADD OPTION TO INCLUDE JSON FILE WITH EXTRA FILTERS
        assert assay in list(crit_dict.keys()), 'Assay type not supported'
        crit_list = crit_dict[assay]
        assay_df = assay_df[(assay_df['File assembly'] == options.gen_assembly)
                 & (assay_df['File Status'] == 'released')
                 & (assay_df['File type'] == 'bed')
                 & (assay_df['File format type']!='bed3+')]
        if options.biosample:
            assay_df = assay_df[(assay_df['Biosample term name'] == options.biosample)]
        if options.crispr:
            assay_df = assay_df[(assay_df['Biosample genetic modifications methods']).isnull().values]

        ass_exp_groups = assay_df.groupby(by='Experiment accession and replicate')
        print("Processing {} {} experiments".format(len(ass_exp_groups), assay))
        for exp_name, df in ass_exp_groups:
            filter_assay(crit_list, df, output_folder, df_filt)
    # Filtered df
    fin_df = pd.concat(df_filt)
    fin_df['label'] = [make_label(row) for _, row in fin_df.iterrows()]
    fin_df['abs_path'] = [get_bed_url(row, output_folder) for _, row in fin_df.iterrows()]
    #filter certain number of rows
    if options.limit:
        fin_df = fin_df.sample(n=options.limit, random_state=options.seed)
    fin_df.to_csv(os.path.join(base_dir, options.biosample+'_filtered_df.csv'))

    with open(os.path.join(base_dir, options.biosample+'_sample_beds.txt'), 'w') as f:
        for i,row in fin_df.iterrows():
            f.write("{}\t{}\n".format(row['label'], row['abs_path']))
    #
    with open(os.path.join(base_dir, options.biosample+'_urls.txt'), 'w') as f:
        for i,row in fin_df.iterrows():
            f.write("{}\n".format(row['File download URL']))



def make_label(df_row):

    '''
    Add label for each row selected:
    Assay_
    Experiment target (IF DNASE REPLACE nan WITH DNA)_
    Biosample term name_
    Experiment accession_
    File accession
    '''
    if df_row['Assay']=='DNase-seq':
        df_row['Experiment target'] = 'DNA'
    label_list = [str(c) for c in [df_row['Assay'], df_row['Experiment target'], df_row['Biosample term name'],
                         df_row['Experiment accession'], df_row['File accession']]]
    return('_'.join(label_list).replace(" ", "-"))

def get_bed_url(df_row, output_dir):
    url = df_row['File download URL']
    output_path = os.path.join(os.path.abspath(output_dir), url.split('/')[-1])
    return(output_path)



def process_priority(c_name, df, output_dir, df_filt):
    """Process a df selection (1 or 2 rows only) based on a set criterion"""
    c_true = df['Output type'] == c_name
    if any(c_true):
        found_c = True
        #preferentially take c_name output type files if any present
        if sum(c_true)==1:
            df_filt.append(df[c_true].iloc[[0]])
        elif sum(c_true)==2:
            df_filt.append(df[c_true].iloc[[0]])
            df_filt.append(df[c_true].iloc[[1]])
        else:
            pass
    else:
        found_c = False
    return found_c

def download_metadata(metadata_url, output_folder):
    print("Downloading metadata.tsv for the project")
    metadata_path = os.path.join(output_folder, 'metadata.tsv')
    urllib.request.urlretrieve(metadata_url, metadata_path)
    metadata = pd.read_csv(metadata_path ,sep='\t')
    return metadata

def filter_assay(crit_list, df, output_dir, df_filt):
        found = False
        for crit in crit_list:
            if not found:
                found = process_priority(crit, df, output_dir, df_filt)


# ################################################################################
# # __main__
################################################################################
if __name__ == '__main__':
    main()
