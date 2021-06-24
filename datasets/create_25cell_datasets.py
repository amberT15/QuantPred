#!/usr/bin/env python
import subprocess
import yaml, os, shutil
# import oyaml as yaml
# parameter grid to loop over
pool_window = [1]
input_size = [3072]
# pool_window = [1]
# input_size = [1024]
# i_w = [(3072, 1)]


# i = 2048
# p = 64

for i in input_size:
    for p in pool_window:
# for _ in [1]:
#     for parameters in i_w:
#         i, p = parameters
        print('Creating dataset using input size %i and pool window %i' % (i, p))
        template_path = r'test_dataset_creation_1pct/template_config.yaml'
        # template_path = r'ATAC_v2/template_config.yaml'
        with open(template_path) as file:
            config = yaml.safe_load(file)

        #update parameters
        config['threshold'] = 2
        config['test_threshold'] = 0
        config['input']['size'] = i
        config['input']['pool'] = p
        config['output']['prefix'] = 'i_%i_w_%i' % (config['input']['size'], config['input']['pool'])
        current_config = 'config.yaml'
        #save as the config file to be used in running the preprocessing
        with open(current_config, 'w') as file:
            documents = yaml.dump(config, file, default_flow_style=False)
        # run dataset generation script with current parameters
        subprocess.call('./bw_to_tfr.sh')
