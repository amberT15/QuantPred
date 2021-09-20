#!/usr/bin/env python
import subprocess
import yaml, os, shutil, sys

# parameter grid to loop over
input_size = 3072
pool_window = 1
dilation_rate = 1
# basset_samplefile = '/mnt/906427d6-fddf-41bf-9ec6-c3d0c37e766f/amber/ATAC/basset_sample_file.tsv'
base_dir = sys.argv[1]
basset_samplefile = sys.argv[2]
basenji_samplefile = sys.argv[3]
limit_to_chroms = 'all'

# for threshold in thresholds:
print('Creating dataset using input size %i and pool window %i' % (input_size, pool_window))
template_path = base_dir+'/template_config.yaml'
# template_path = r'ATAC_v2/template_config.yaml'
with open(template_path) as file:
    config = yaml.safe_load(file)

#update parameters
if 'chroms' not in config.keys():
  config['chroms']={}
config['chroms']['only'] = limit_to_chroms
if 'all' != config['chroms']['only'] and 'none' != config['chroms']['only']:
    config['chroms']['valid'] = 0
    config['chroms']['test'] = 1.0
else:
    config['chroms']['valid'] = 'chr9'
    config['chroms']['test'] = 'chr8'
config['input']['downsample'] = dilation_rate
config['samplefile']['basset'] = basset_samplefile
config['samplefile']['basenji'] = basenji_samplefile
config['threshold'] = 2
config['test_threshold'] = -1
config['input']['size'] = input_size
# config['input']['pool'] = pool_window
config['output']['dir'] = os.path.join(base_dir)
config['output']['prefix'] = 'i_%i_w_%i' % (config['input']['size'], config['input']['pool'])
current_config = 'config.yaml'
#save as the config file to be used in running the preprocessing
with open(current_config, 'w') as file:
    documents = yaml.dump(config, file, default_flow_style=False)
# run dataset generation script with current parameters
subprocess.call('./bw_to_tfr.sh')
