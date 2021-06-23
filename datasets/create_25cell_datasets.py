#!/usr/bin/env python
import subprocess
import yaml, os, shutil

# parameter grid to loop over
input_size = [3072, 2048]
# pool_window = [1]
# input_size = [1024]
# i_w = [(2048, 1)]


# i = 2048
# p = 64

base_dir = '/mnt/1a18a49e-9a31-4dbf-accd-3fb8abbfab2d/shush/4grid_atac/complete'


for i in input_size:
    for basset_samplefile in ['random', '/mnt/906427d6-fddf-41bf-9ec6-c3d0c37e766f/amber/ATAC/basset_sample_file.tsv']:
        p = 1
        print('Creating dataset using input size %i and pool window %i' % (i, p))
        template_path = os.path.join(base_dir, 'template_config.yaml')
        print(template_path)
        # template_path = r'ATAC_v2/template_config.yaml'
        with open(template_path) as file:
            config = yaml.safe_load(file)

        #update parameters
        config['input']['size'] = i
        config['input']['pool'] = p
        config['samplefile']['basset'] = basset_samplefile
        if basset_samplefile=='random':
            config['output']['dir'] = os.path.join(base_dir, 'random_chop')
            config['threshold'] = 2
            config['test_threshold'] = -1

        else:
            config['output']['dir'] = os.path.join(base_dir, 'peak_centered')
            config['threshold'] = -1
            config['test_threshold'] = -1

        config['output']['prefix'] = 'i_%i_w_%i' % (config['input']['size'], config['input']['pool'])
        current_config = 'config.yaml'
        #save as the config file to be used in running the preprocessing
        with open(current_config, 'w') as file:
            documents = yaml.dump(config, file, default_flow_style=False)
        # run dataset generation script with current parameters
        subprocess.call('./bw_to_tfr.sh')
