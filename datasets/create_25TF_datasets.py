#!/usr/bin/env python
import subprocess
import yaml, os, shutil

# parameter grid to loop over
pool_window = [1, 32, 64, 128, 256]
input_size = [1024, 2048, 3072]
# i = 2048
# p = 64

for i in input_size:
    for p in pool_window:
        print('Creating dataset using input size %i and pool window %i' % (i, p))
        template_path = r'top25/template_config.yaml'


        with open(template_path) as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        #update parameters
        config['input']['size'] = i
        config['input']['pool'] = p
        config['output']['prefix'] = 'i_%i_w_%i' % (config['input']['size'], config['input']['pool'])
        current_config = 'config.yaml'
        #save as the config file to be used in running the preprocessing
        with open(current_config, 'w') as file:
            documents = yaml.dump(config, file)
        # run dataset generation script with current parameters
        subprocess.call('./bw_to_tfr.sh')
