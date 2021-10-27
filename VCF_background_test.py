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
import modelzoo
import explain
import tfr_evaluate
import glob

#get model that we want to run VCF on
all_run_metadata = []
# model_path_pair = {'Basenji 128':'paper_runs/basenji/augmentation_basenji/*'}
model_path_pair = {'BPNet 1':'paper_runs/bpnet/augmentation_48/*'}
for run_path in glob.glob(list(model_path_pair.values())[0]):
    all_run_metadata.append(tfr_evaluate.get_run_metadata(run_path))
all_run_metadata = pd.concat(all_run_metadata)
all_run_metadata['dataset'] = ['random_chop' if 'random' in data_dir else 'peak_centered' for data_dir in all_run_metadata['data_dir'].values]
model_paths = []
bin_size = int(list(model_path_pair.keys())[0].split(' ')[-1])
for i, df in all_run_metadata[all_run_metadata['bin_size']==1].groupby(['crop', 'rev_comp', 'dataset']):
    assert df.shape[0] == 3, 'mip'""
    model_paths.append(df.iloc[0]['run_dir'])

# load in dataset that conatin both dsQTL sites and control SNP sites    
vcf_data = './datasets/VCF/dsQTL_onehot.h5'
f =  h5py.File(vcf_data, "r")
alt_3k = f['alt'][()]
ref_3k = f['ref'][()]
f.close()

control_data = './datasets/VCF/negative_onehot.h5'
f =  h5py.File(control_data, "r")
c_alt_3k = f['alt'][()]
c_ref_3k = f['ref'][()]
f.close()    
    
#loop through models and run function   
for model_path in model_paths:
    #load model
    model = modelzoo.load_model(model_path,compile = True)
    
    #run robust vcf on both QTL and control
    vcf_diff = explain.vcf_robust(ref_3k,alt_3k,model)
    background_diff = explain.vcf_robust(c_ref_3k,c_alt_3k,model)
    vcf_diff = np.concatenate(vcf_diff)
    background_diff = np.concatenate(background_diff)

    
    #decide ouput directory and file name
    tmp_df = all_run_metadata[all_run_metadata['run_dir']==model_path]
    vcf_output_path = './datasets/VCF/VCF_results' +'/'+tmp_df['model_fn'].values[0]+'_'+ str(tmp_df['crop'].values[0]) +'_'+str(tmp_df['rev_comp'][0])+'_'+str(tmp_df['dataset'].values[0])+'.h5'
    
    #write output file
    h5_dataset = h5py.File(vcf_output_path, 'w')
    h5_dataset.create_dataset('vcf_diff', data=vcf_diff)
    h5_dataset.create_dataset('background_diff',data = background_diff)
    h5_dataset.close()