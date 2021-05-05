#!/usr/bin/env python
import os

all_data = '/home/shush/profile/QuantPred/datasets/top25/'
output_file = 'commands_grid1.txt'
epoch_n = 100
# define grid points

# inputs
inputs_all = [os.path.join(all_data, foldername) for foldername in os.listdir(all_data) if 'w_' in foldername]
inputs_basenji = [os.path.join(all_data, foldername) for foldername in os.listdir(all_data) if 'w_32' in foldername]
inputs_bpnet = [os.path.join(all_data, foldername) for foldername in os.listdir(all_data) if 'w_32' in foldername]

# model


# losses

losses = ['poisson', 'mse', 'pearsonr', 'fft_mse', 'multinomial_nll']

cmd = []
# loop and save commands to a list
output_dir = '/home/shush/profile/QuantPred/datasets/top25/grid1'
for input in inputs_basenji:
    for loss in losses:
        cmd.append('./train.py {} basenji_small {} -e {} -o {}'.format(input, loss, epoch_n, output_dir))

# add more loops here
for input in inputs_bpnet:
    for loss in losses:
        cmd.append('./train.py {} bpnet {} -e {} -o {}'.format(input, loss, epoch_n, output_dir))

# add more loops here
for input in inputs_all:
    for loss in losses:
        cmd.append('./train.py {} custom_lstm {} -e {} -o {}'.format(input, loss, epoch_n, output_dir))

# save list to file
with open(output_file, 'w') as f:
    for command in cmd:
        f.write("%s\n" % command)
