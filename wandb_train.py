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
from tensorflow.keras.callbacks import ModelCheckpoint
import dataset
# import trainer_class
from custom_fit import *
import time
import wandb
from wandb.keras import WandbCallback
from wandb_callbacks import *

def fit_robust(model_name_str, loss_type_str, window_size, bin_size, data_dir,
               num_epochs=50, batch_size=64, shuffle=True, output_dir='.',
               metrics=['mse','pearsonr', 'poisson'], mix_epoch=50,  es_start_epoch=50,
               l_rate=0.004, es_patience=6, es_metric='val_loss',
               es_criterion='min', lr_decay=0.3, lr_patience=10,
               lr_metric='val_loss', lr_criterion='min', verbose = True, **kwargs):

  optimizer = tf.keras.optimizers.Adam(learning_rate=l_rate)
  model = eval(model_name_str) # get model function from model zoo
  output_len = window_size // bin_size

  loss = eval(loss_type_str)() # get loss from loss.py
  trainset = util.make_dataset(data_dir, 'train', util.load_stats(data_dir))
  validset = util.make_dataset(data_dir, 'valid', util.load_stats(data_dir))
  json_path = os.path.join(data_dir, 'statistics.json')
  with open(json_path) as json_file:
    params = json.load(json_file)
  model = model((window_size, 4),(output_len, params['num_targets']), **kwargs)
  print(model.summary())
  train_seq_len = params['train_seqs']

  # create trainer class
  trainer = RobustTrainer(model, loss, optimizer, window_size, bin_size, metrics)

  # set up learning rate decay
  trainer.set_lr_decay(decay_rate=lr_decay, patience=lr_patience, metric=lr_metric, criterion=lr_criterion)
  trainer.set_early_stopping(patience=es_patience, metric=es_metric, criterion=es_criterion)

  # train model
  for epoch in range(num_epochs):
    sys.stdout.write("\rEpoch %d \n"%(epoch+1))

    #Robust train with crop and bin
    # print('blaanot')
    trainer.robust_train_epoch(trainset, window_size, bin_size,num_step=train_seq_len//batch_size+1,batch_size = batch_size)

    # validation performance
    trainer.robust_evaluate('val', validset,window_size, bin_size, batch_size=batch_size, verbose=verbose)

    # check early stopping
    if epoch >= es_start_epoch:

      # check learning rate decay
      trainer.check_lr_decay('val')

      if trainer.check_early_stopping('val'):
        print("Patience ran out... Early stopping.")
        break
    # Logging with W&B
    current_hist = trainer.get_current_metrics('train')
    wandb.log(trainer.get_current_metrics('val', current_hist))


  # compile history
  history = trainer.get_metrics('train')
  history = trainer.get_metrics('val', history)
  model.save(os.path.join(output_dir, "best_model.h5"))
  # print(history)
  return history



def train_config(config=None):

  with wandb.init(config=config) as run:

    config = wandb.config
    print(config.data_dir)
    print(config.l_rate)


    history = fit_robust(config.model_fn, config.loss_fn,
                       config.window_size, config.bin_size, config.data_dir,
                       l_rate=config.l_rate, num_epochs=config.epochN,
                       output_dir=wandb.run.dir)


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

def main():
	sweep_id = 'toneyan/tune_bpnet/0cvlt5ho'
	wandb.login()
	wandb.agent(sweep_id, train_config, count=18)

# __main__
################################################################################
if __name__ == '__main__':
    main()




# def train(data_dir, model_name_str, loss_type_str, out_dir='.',
#           n_epochs=2, earlystop_p=6, l_rate=0.004, save_with_wandb=True):
#     # create output folder if not present
#     if not os.path.isdir(out_dir):
#       os.mkdir(out_dir)
#
#     batch_size = 64
#     #load data
#     train_data, eval_data, test_data = load_data(data_dir, batch_size)
#
#     # read json statistics file
#     json_path = os.path.join(data_dir, 'statistics.json')
#     with open(json_path) as json_file:
#         params = json.load(json_file)
#     input_size = params['seq_length']
#     num_targets = params['num_targets']
#     n_seqs = params['valid_seqs']
#     output_length = int(params['seq_length']/params['pool_width']) # needed for binned data
#
#     print('Input size is {}, number of TFs is {}'.format(input_size, num_targets))
#     model_fn = eval(model_name_str) # get model function from model zoo
#     loss_fn = eval(loss_type_str) # get loss from loss.py
#     seqnn_model = model_fn((input_size, 4), (output_length, num_targets))
#     # define paths to save outputs
#     data_folder = os.path.basename(os.path.normpath(data_dir))
#     prefix = '{}_{}_{}'.format(data_folder, model_name_str, loss_type_str)
#     out_model_path = os.path.join(out_dir, 'model_'+prefix+'.h5')
#
#     # initialize trainer
#     seqnn_trainer = trainer_class.Trainer(train_data,
#                                 eval_data, out_model_path, n_epochs, loss_fn, batch_size,
#                                 learning_rate=l_rate, patience=earlystop_p,
#                                 optimizer='adam')
#
#     # compile model
#     seqnn_trainer.compile(seqnn_model)
#     # define wandb callbacks
#     if save_with_wandb:
#         reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
#                                                              factor=0.2, # TODO .2
#                                                              patience=3, # 3
#                                                              min_lr=1e-7,
#                                                              mode='min',
#                                                              verbose=1)
#         # early_stop = trainer_class.EarlyStoppingMin(monitor='val_pearsonr', mode='max', verbose=1,
#         #                    patience=earlystop_p)
#         wandb_callback = WandbCallback()
#         save_callback = ModelCheckpoint(os.path.join(wandb.run.dir, "best_model.tf"), save_best_only=True, mode='max',
#         monitor='val_pearsonr', verbose=1)
#         callbacks = [reduce_lr, wandb_callback, save_callback]
#     else:
#         callbacks = []
#     start = time.time()
#     history = seqnn_trainer.fit_keras(seqnn_model, callbacks)
#     end = time.time()
#     print('Training duration: {}min'.format(str(round((end-start)/60))))
#     return history.history
    # print('Saving outputs using prefix ' + prefix)
    # out_pred_path = os.path.join(options.out_dir, 'pred_'+prefix+'.h5')
    #
    #
    #
    #
    # test_y = util.tfr_to_np(test_data[0].dataset, 'y', (params['test_seqs'], output_length, params['num_targets']))
    # test_x = util.tfr_to_np(test_data[0].dataset, 'x', (params['test_seqs'], params['seq_length'], 4))
    # test_pred = seqnn_model.predict(test_x)
    # hf = h5py.File(out_pred_path, 'w')
    # hf.create_dataset('test_x', data=test_x)
    # hf.create_dataset('test_y', data=test_y)
    # hf.create_dataset('test_pred', data=test_pred)
    # hf.close()
