#!/usr/bin/env python
from optparse import OptionParser
import os
import urllib.request
import pandas as pd
import sys
import utils

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

def process(filtered_row, output_dir):
    label = make_label(filtered_row) # get the label
    # get the output path and url
    output_path, url = get_bed_url(filtered_row, output_dir)
    # return 'label \t output path' to be added to a txt file
    return('\t'.join([label,output_path]), url)

def process_priority(c_name, df, txt_lines, output_dir,url_txt):
    """Process a df selection (1 or 2 rows only) basd on a set criterion"""
    c_true = df['Output type'] == c_name
    if any(c_true):
        found_c = True
        #preferentially take c_name output type files if any present
        if sum(c_true)==1:
            bed_text, url = process(df[c_true].iloc[[0]], output_dir)
            txt_lines.append(bed_text)
            url_txt.append(url)
        elif sum(c_true)==2:
            bed_text, url = process(df[c_true].iloc[[0]], output_dir)
            txt_lines.append(bed_text)
            url_txt.append(url)
            bed_text, url = process(df[c_true].iloc[[1]], output_dir)
            txt_lines.append(bed_text)
            url_txt.append(url)
        else:
            pass
    else:
        found_c = False
    return found_c

def download_metadata(metadata_url, output_folder):
    print("Downloading metadata.tsv for the project")
    metadata_path = os.path.join(output_folder, 'metadata.tsv')
    #TODO: uncomment
    # urllib.request.urlretrieve(metadata_url, metadata_path)
    metadata = pd.read_csv(metadata_path ,sep='\t')
    return metadata

def filter_dnase(df, txt_lines, output_dir, url_txt):
    '''
    df = one DNase-seq experiment dataframe with bed files only and genome filtered
    txt_lines = list of lines of label and path of the corresponding file
    '''
    c1 = 'peaks'
    c1_found = process_priority(c1, df, txt_lines, output_dir,url_txt)

def filter_hist(df, txt_lines, output_dir, url_txt):
    c1 = 'replicated peaks'
    c2 = 'pseudo-replicated peaks'
    c1_found = process_priority(c1, df, txt_lines, output_dir, url_txt)
    if not c1_found:
        c2_found = process_priority(c2, df, txt_lines, output_dir, url_txt)

def filter_tf(df, txt_lines, output_dir, url_txt):
    c1 = 'conservative IDR thresholded peaks'
    c2 = 'optimal IDR thresholded peaks'
    c3 = 'pseudoreplicated IDR thresholded peaks'
    c1_found = process_priority(c1, df, txt_lines, output_dir, url_txt)
    ##UNCOMMENT IF OK TO INCLUDE WORSE QUALITY DATA
    # if not c1_found:
    #     c2_found = process_priority(c2, df, txt_lines, output_dir, url_txt)
    #     if not c2_found:
    #         process_priority(c3, df, txt_lines, output_dir, url_txt)

def main():
    usage = 'usage: %prog [options] <files_path> <output_folder>'
    parser = OptionParser(usage)
    parser.add_option('-g', dest='gen_assembly',
      default='hg19', type='str',
      help='Genome assembly [Default: %default]')
    parser.add_option('-b', dest='biosample',
      default='', type='str',
      help='Biosample to search and keep [Default: %default]')
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
    metadata['Experiment accession'] = metadata['Experiment accession']+'_'+ metadata['Biological replicate(s)']

    assay_groups = metadata.groupby(by='Assay')

    assay_filter_dict = {'TF ChIP-seq':filter_tf, 'Histone ChIP-seq':filter_hist, 'DNase-seq':filter_dnase}
    all_lines = []
    url_txt = []
    utils.make_directory(output_folder)
    for assay, assay_df in assay_groups:
        # TODO:ADD OPTION TO INCLUDE JSON FILE WITH EXTRA FILTERS
        assert assay in list(assay_filter_dict.keys()), 'Assay type not supported'
        filter_func = assay_filter_dict[assay]
        assay_df = assay_df[(assay_df['File assembly'] == options.gen_assembly)
                 & (assay_df['File Status'] == 'released')
                 & (assay_df['File type'] == 'bed')
                 & (assay_df['File format type']!='bed3+')]
        if options.biosample:
            assay_df = assay_df[(assay_df['Biosample term name'] == options.biosample)]
        ass_exp_groups = assay_df.groupby(by='Experiment accession')
        print("Processing {} {} experiments".format(len(ass_exp_groups), assay))
        for exp_name, df in ass_exp_groups:
            filter_func(df, all_lines, output_folder, url_txt)


    with open(os.path.join(base_dir, 'sample_beds.txt'), 'w') as f:
        for item in all_lines:
            f.write("%s\n" % item)

    with open(os.path.join(base_dir, 'urls.txt'), 'w') as f:
        for item in url_txt:
            f.write("%s\n" % item.strip())


# ################################################################################
# # __main__
################################################################################
if __name__ == '__main__':
    main()
