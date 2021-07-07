#!/usr/bin/env python
import subprocess
import yaml, os, shutil
# import oyaml as yaml
# parameter grid to loop over
pool_window = 1
input_size = 2048
# pool_window = [1]
# input_size = [1024]
# i_w = [(3072, 1)]
base_dir = '0707_4grid'
# basset_samplefile_values = ['Random', '/mnt/906427d6-fddf-41bf-9ec6-c3d0c37e766f/amber/ATAC/basset_sample_file.tsv']
basset_samplefile_values = ['Random']
output_subdir = [os.path.join(base_dir, 'lite', d) for d in ['random_chop', 'peak_centered']]
# i = 2048
# p = 64

for b, basset_samplefile in enumerate(basset_samplefile_values):

    print('Creating dataset using input size %i and pool window %i' % (input_size, pool_window))
    template_path = base_dir+'/template_config.yaml'
    # template_path = r'ATAC_v2/template_config.yaml'
    with open(template_path) as file:
        config = yaml.safe_load(file)

    #update parameters
    config['samplefile']['basset'] = basset_samplefile
    config['threshold'] = 2
    config['test_threshold'] = -1
    config['input']['size'] = input_size
    config['input']['pool'] = pool_window
    config['output']['dir'] = output_subdir[b]
    config['output']['prefix'] = 'i_%i_w_%i' % (config['input']['size'], config['input']['pool'])
    current_config = 'config.yaml'
    #save as the config file to be used in running the preprocessing
    with open(current_config, 'w') as file:
        documents = yaml.dump(config, file, default_flow_style=False)
    # run dataset generation script with current parameters
    subprocess.call('./bw_to_tfr.sh')
