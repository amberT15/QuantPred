#!/usr/bin/env python
import os

all_data = '/home/shush/profile/QuantPred/datasets/top25/'
output_file = 'commands.txt'
epoch_n = 2
# define grid points

# inputs
inputs = [os.path.join(all_data, foldername) for foldername in os.listdir(all_data) if 'w_32' in foldername]


# model

model_names = ['basenji_small']

# losses

losses = ['poisson', 'mse', 'pearsonr', 'fft_mse', 'multinomial_nll']

cmd = []
# loop and save commands to a list
output_dir = 'test_dir'
for input in inputs:
    for model in model_names:
        for loss in losses:
            cmd.append('./train.py {} {} {} -e {} -o {}'.format(input, model, loss, epoch_n, output_dir))

# add more loops here




# save list to file
with open(output_file, 'w') as f:
    for command in cmd:
        f.write("%s\n" % command)
