#!/usr/bin/env python

import json
import os
import h5py
import sys
import util
from optparse import OptionParser
from natsort import natsorted
import numpy as np
import tensorflow as tf
from modelzoo import *
from loss import *
import dataset
import trainer_class
import time

def main():
    usage = 'usage: %prog [options] <data_dir> <model name> <loss type>'
    parser = OptionParser(usage)
    parser.add_option('-o', dest='out_dir', default='.', help='Output where model and pred will be saved')
    parser.add_option('--notest', default=True, action='store_false', dest='save_test', help='Make and save test set predictions')
    parser.add_option('-e', dest='n_epochs', default=100, type='int', help='N epochs [Default: %default]')
    parser.add_option('-p', dest='earlystop_p', default=6, type='int', help='Early stopping patience [Default: %default]') # TODO: 6
    parser.add_option('-l', dest='l_rate', default=0.001, type='float', help='learning rate [Default: %default]')
    (options,args) = parser.parse_args()

    if len(args) != 3:
        parser.error('Must provide data dir, model architecture name, and loss')
    else:
        data_dir = args[0]
        model_name_str = args[1]
        loss_type_str = args[2]

    # create output folder if not present
    if not os.path.isdir(options.out_dir):
      os.mkdir(options.out_dir)

    batch_size = 64
    #load data
    train_data, eval_data, test_data = load_data(data_dir, batch_size)

    # read json statistics file
    json_path = os.path.join(data_dir, 'statistics.json')
    with open(json_path) as json_file:
        params = json.load(json_file)
    input_size = params['seq_length']
    num_targets = params['num_targets']
    n_seqs = params['valid_seqs']
    output_length = int(params['seq_length']/params['pool_width']) # needed for binned data

    print('Input size is {}, number of TFs is {}'.format(input_size, num_targets))
    model_fn = eval(model_name_str) # get model function from model zoo
    # if loss_fn == 'poisson_direct':
    #     loss_fn =
    loss_fn = eval(loss_type_str)() # get loss from loss.py
    seqnn_model = model_fn((input_size, 4), (output_length, num_targets))
    # define paths to save outputs
    data_folder = os.path.basename(os.path.normpath(data_dir))
    prefix = '{}_{}_{}'.format(data_folder, model_name_str, loss_type_str)
    out_model_path = os.path.join(options.out_dir, 'model_'+prefix+'.h5')

    # initialize trainer
    seqnn_trainer = trainer_class.Trainer(train_data,
                                eval_data, out_model_path, options.n_epochs, loss_fn, batch_size,
                                learning_rate=options.l_rate, patience=options.earlystop_p,
                                optimizer='adam')

    # compile model
    seqnn_trainer.compile(seqnn_model)
    start = time.time()
    history = seqnn_trainer.fit_keras(seqnn_model)
    hist_df = pd.DataFrame(history.history)
    # or save to csv:
    hist_csv_file = os.path.join(options.out_dir, 'history.csv')
    hist_df.to_csv(f)
    end = time.time()
    print('Training duration: {}min'.format(str(round((end-start)/60))))
    if options.save_test:
        print('Saving outputs using prefix ' + prefix)
        out_pred_path = os.path.join(options.out_dir, 'pred_'+prefix+'.h5')


        sts = util.load_stats(data_dir)
        testset = util.make_dataset(data_dir, 'test', sts, coords=True)
        # initialize inputs and outputs
        seqs_1hot = []
        targets = []
        coords_list = []
        # collect inputs and outputs
        for coord, x, y in testset:
        # sequence
            seq_raw, targets_raw = x,y

            seq = seq_raw.numpy()
            seqs_1hot.append(seq)

            # targets
            targets1 = targets_raw.numpy()
            targets.append(targets1)

            # coords
            coords_list.append(coord)

        seqs_all = np.concatenate((seqs_1hot))
        targets_all = np.concatenate(targets)
        coords_str_list = [[str(c).strip('b\'chr').strip('\'') for c in coords.numpy()] for coords in coords_list]
        nonsplit_x_y = [item for sublist in coords_str_list for item in sublist]

        coords_all = np.array([replace_all(item) for item in nonsplit_x_y])
        coords_all = coords_all.astype(np.int)

        test_pred = model(tf.convert_to_tensor(seqs_all))
        hf = h5py.File(out_pred_path, 'w')
        hf.create_dataset('test_x', data=seqs_all)
        hf.create_dataset('test_y', data=targets_all)
        hf.create_dataset('coords', data=coords_all)
        hf.create_dataset('test_pred', data=test_pred)
        hf.close()

def replace_all(text):
    dic = {'X': '23', 'Y': '24'}
    for i, j in dic.items():
        text = text.replace(i, j)
    return text.split('_')

def load_data(data_dir, batch_size):
  '''
  Load TFrecords dataset and return training and validation sets
  '''
  # read datasets
  train_data = []
  eval_data = []
  test_data = []

  # load train data
  train_data.append(dataset.SeqDataset(data_dir,
  split_label='train',
  batch_size=batch_size,
  mode='train',
  tfr_pattern=None))

  # load eval data
  eval_data.append(dataset.SeqDataset(data_dir,
  split_label='valid',
  batch_size=batch_size,
  mode='eval',
  tfr_pattern=None))


  # load eval data
  test_data.append(dataset.SeqDataset(data_dir,
  split_label='test',
  batch_size=batch_size,
  mode='eval',
  tfr_pattern=None))
  return (train_data, eval_data, test_data)


# __main__
################################################################################
if __name__ == '__main__':
    main()
