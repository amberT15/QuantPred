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
    (options, args) = parser.parse_args()
    if len(args) != 2:
        parser.error('Must provide file list from ENCODE and output folder')
    else:
        files_path = args[0]
        output_folder = args[1]

    base_dir = utils.get_parent(files_path)
    with open(files_path, "r") as file:
        metadata_url = file.readline()[1:-2] #remove " before and after url

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
    all_lines = []
    url_txt = []
    utils.make_directory(output_folder)
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
            filter_assay(crit_list, df, all_lines, output_folder, url_txt, df_filt)
    fin_df = pd.concat(df_filt)
    fin_df.to_csv(os.path.join(base_dir, options.biosample+'_filtered_df.csv'))

    with open(os.path.join(base_dir, options.biosample+'_sample_beds.txt'), 'w') as f:
        for item in all_lines:
            f.write("%s\n" % item)

    with open(os.path.join(base_dir, options.biosample+'_urls.txt'), 'w') as f:
        for item in url_txt:
            f.write("%s\n" % item.strip())



def make_label(df_row):

    '''
    Add label for each row selected:
    Assay_
    Experiment target (IF DNASE REPLACE nan WITH DNA)_
    Biosample term name_
    Experiment accession_
    File accession
    '''
    if df_row['Assay'].iloc[0]=='DNase-seq':
        df_row['Experiment target'] = 'DNA'
    label_list = [str(c.iloc[0]) for c in [df_row['Assay'], df_row['Experiment target'], df_row['Biosample term name'],
                         df_row['Experiment accession'], df_row['File accession']]]
    return('_'.join(label_list).replace(" ", "-"))

def get_bed_url(df_row, output_dir):
    url = df_row['File download URL'].iloc[0]
    output_path = os.path.join(output_dir, url.split('/')[-1])
    return(output_path, url)



def process_priority(c_name, df, txt_lines, output_dir,url_txt, df_filt):
    """Process a df selection (1 or 2 rows only) based on a set criterion"""
    def process(filtered_row, output_dir):
        label = make_label(filtered_row) # get the label
        # get the output path and url
        output_path, url = get_bed_url(filtered_row, output_dir)
        bed_text = '\t'.join([label,output_path])
        txt_lines.append(bed_text)
        url_txt.append(url)
        df_filt.append(filtered_row)

    c_true = df['Output type'] == c_name
    if any(c_true):
        found_c = True
        #preferentially take c_name output type files if any present
        if sum(c_true)==1:
            process(df[c_true].iloc[[0]], output_dir)

        elif sum(c_true)==2:
            process(df[c_true].iloc[[0]], output_dir)
            process(df[c_true].iloc[[1]], output_dir)

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

def filter_assay(crit_list, df, txt_lines, output_dir, url_txt, df_filt):
        found = False
        for crit in crit_list:
            if not found:
                found = process_priority(crit, df, txt_lines, output_dir, url_txt, df_filt)


# ################################################################################
# # __main__
################################################################################
if __name__ == '__main__':
    main()
