#!/usr/bin/env python
import subprocess
import yaml, os, shutil

# parameter grid to loop over
# pool_window = [1, 32, 64, 128, 256]
# input_size = [1024, 2048, 3072]

p = 64
i = 512

template_path = r'top25/template_config.yaml'


with open(template_path) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
#update parameters
config['input']['size'] = i
config['input']['pool'] = p
config['output']['prefix'] = 'i_%i_w_%i' % (config['input']['size'], config['input']['pool'])
# loops
# for p in pool_window:
#     for i in input_size:
current_config = 'config.yaml'
with open(current_config, 'w') as file:
    documents = yaml.dump(config, file)
subprocess.call('./bw_to_tfr.sh')
