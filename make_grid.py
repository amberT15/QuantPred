#!/usr/bin/env python
import os

all_data = '/home/shush/profile/QuantPred/datasets/top25/'
output_file = 'test.txt'
# define grid points

# inputs
inputs = [os.path.join(all_data, foldername) for foldername in os.listdir(all_data) if 'w_32' in foldername]
print('Add inputs in directories')
print(inputs)

# model

model_names = ['basenji_small']

# losses

losses = ['poisson', 'mse', 'pearsonr', 'fft_mse_loss']

cmd = []
# loop and save commands to a list
for input in inputs:
    for model in model_names:
        for loss in losses:
            cmd.append('./train.py {} {} {} -e 1 -o test_run'.format(input, model, loss))


# save list to file
with open(output_file, 'a') as f:
    for command in cmd:
        f.write("%s\n" % command)
