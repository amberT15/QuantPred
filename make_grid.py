#!/usr/bin/env python
import os

all_data = '/home/shush/profile/QuantPred/datasets/ATAC_v2/'
output_file = 'commands_grid5.txt'
epoch_n = 100
# define grid points

# inputs
inputs_all = [os.path.join(all_data, foldername) for foldername in os.listdir(all_data) if 'w_' in foldername]
inputs_basenji = [os.path.join(all_data, foldername) for foldername in os.listdir(all_data) if 'w_32' in foldername]
inputs_bpnet = [os.path.join(all_data, foldername) for foldername in os.listdir(all_data) if 'w_1' in foldername and 'w_128' not in foldername]

# model


# losses

losses = ['poisson', 'mse', 'basenjipearsonr', 'fftmse', 'multinomialnll']
# losses = ['poisson']
cmd = []
# loop and save commands to a list
output_dir = '/home/shush/profile/QuantPred/datasets/ATAC_v2/grid5'
# output_dir = '/home/shush/profile/QuantPred/testrun'

# add loops here
for input in inputs_all:
    for loss in losses:
        if loss != 'poisson':
            cmd.append('./train.py {} bpnet {} -e {} -o {}'.format(input, loss, epoch_n, output_dir))

for input in inputs_all:
    for loss in losses:
        if 'w_1' in input:
            cmd.append('./train.py {} basenjiw1 {} -e {} -o {} '.format(input, loss, epoch_n, output_dir))

        else:
            cmd.append('./train.py {} basenjimod {} -e {} -o {} '.format(input, loss, epoch_n, output_dir))



# add more loops here
# for input in inputs_all:
#     for loss in losses:
#         cmd.append('./train_bas.py {} lstm {} -e {} -o {}'.format(input, loss, epoch_n, output_dir))

# save list to file
with open(output_file, 'w') as f:
    for command in cmd:
        f.write("%s\n" % command)
