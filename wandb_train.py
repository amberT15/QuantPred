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
import time
import wandb
from wandb.keras import WandbCallback
from wandb_callbacks import *
#import bpnet_original_fit as bpnet_fit
import custom_fit

def fit_robust(model_name_str, loss_type_str, window_size, bin_size, data_dir,
               config={}, num_epochs=100, batch_size=64, shuffle=True, output_dir='.',
               metrics=['mse','pearsonr', 'poisson'], mix_epoch=50,  es_start_epoch=50,
               l_rate=0.001, es_patience=6, es_metric='val_loss',
               es_criterion='min', lr_decay=0.3, lr_patience=10,
               lr_metric='val_loss', lr_criterion='min', verbose = True,
               log_wandb=True,rev_comp = True, crop_window = True,
               record_test=False, alpha=False):

  if '2048' in data_dir:
      rev_comp = False
      crop_window = True



  if not os.path.isdir(output_dir):
      os.mkdir(output_dir)

  optimizer = tf.keras.optimizers.Adam(learning_rate=l_rate)
  model = eval(model_name_str) # get model function from model zoo
  output_len = window_size // bin_size

  if alpha:
      loss = eval(loss_type_str)(alpha=alpha) # get loss from loss.py
  else:
      loss = eval(loss_type_str)()



  trainset = util.make_dataset(data_dir, 'train', util.load_stats(data_dir))
  validset = util.make_dataset(data_dir, 'valid', util.load_stats(data_dir))

  json_path = os.path.join(data_dir, 'statistics.json')
  with open(json_path) as json_file:
    params = json.load(json_file)

  model = model((window_size, 4),(output_len, params['num_targets']), wandb_config=config)

  if not model:
    raise BaseException('Fatal filter N combination!')


  print(model.summary())
  train_seq_len = params['train_seqs']
  if model_name_str == 'ori_bpnet':
  # create trainer class
    trainer =custom_fit.RobustTrainer(model, loss, optimizer, window_size, bin_size, metrics,ori_bpnet_flag = True)
  else:
    trainer = custom_fit.RobustTrainer(model, loss, optimizer, window_size, bin_size, metrics, ori_bpnet_flag = False)

  # set up learning rate decay
  trainer.set_lr_decay(decay_rate=lr_decay, patience=lr_patience, metric=lr_metric, criterion=lr_criterion)
  trainer.set_early_stopping(patience=es_patience, metric=es_metric, criterion=es_criterion)

  # train model
  for epoch in range(num_epochs):
    sys.stdout.write("\rEpoch %d \n"%(epoch+1))

    #Robust train with crop and bin
    # print('blaanot')
    trainer.robust_train_epoch(trainset, window_size, bin_size,
                                num_step=train_seq_len//batch_size+1,
                                batch_size = batch_size,
                                rev_comp = rev_comp,
                                crop_window = crop_window)

    # validation performance
    trainer.robust_evaluate('val', validset,window_size, bin_size,
                            batch_size=batch_size, verbose=verbose,
                            crop_window = crop_window)

    # check early stopping
    if epoch >= es_start_epoch:

      # check learning rate decay
      trainer.check_lr_decay('val')

      if trainer.check_early_stopping('val'):
        print("Patience ran out... Early stopping.")
        break
    if log_wandb:
        # Logging with W&B
        current_hist = trainer.get_current_metrics('train')
        wandb.log(trainer.get_current_metrics('val', current_hist))

  # compile history
  history = trainer.get_metrics('train')
  history = trainer.get_metrics('val', history)
  model.save(os.path.join(output_dir, "best_model.h5"))
  # print(history)

  if record_test==True:
      testset = util.make_dataset(data_dir, 'test', util.load_stats(data_dir), coords=True)
      out_pred_path = os.path.join(output_dir, 'pred.h5')
      # test_y = util.tfr_to_np(testset, 'y', (params['test_seqs'], window_size, params['num_targets']))
      # test_x = util.tfr_to_np(testset, 'x', (params['test_seqs'], params['seq_length'], 4))
      # initialize inputs and outputs
      seqs_1hot = []
      targets = []
      coords_list = []
      # collect inputs and outputs
      for coord, x, y in testset:
        # sequence
        seq_raw, targets_raw = custom_fit.valid_window_crop(x,y,window_size,bin_size)

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

      coords_all = np.array([util.replace_all(item) for item in nonsplit_x_y])
      coords_all = coords_all.astype(np.int)

      test_pred = model(tf.convert_to_tensor(seqs_all))
      hf = h5py.File(out_pred_path, 'w')
      hf.create_dataset('test_x', data=seqs_all)
      hf.create_dataset('test_y', data=targets_all)
      hf.create_dataset('coords', data=coords_all)
      if model_name_str == 'ori_bpnet':
        hf.create_dataset('pred_profile', data=np.array(test_pred[0]))
        hf.create_dataset('pred_count', data=np.array(test_pred[1]))
      else:
        hf.create_dataset('test_pred', data=test_pred)

      hf.close()


  return history

def train_config(config=None):

  with wandb.init(config=config) as run:

    config = wandb.config
    print(config.data_dir)
    print(config.l_rate)



    history = fit_robust(config.model_fn, config.loss_fn,
                       config.window_size, config.bin_size, config.data_dir,
                       l_rate=config.l_rate, num_epochs=config.epochN,
                       output_dir=wandb.run.dir, rev_comp = True,
                       crop_window = True)


def main():
  exp_id = sys.argv[1]
  exp_n = sys.argv[2]
  if 'sweeps' in exp_id:
      exp_id = '/'.join(exp_id.split('/sweeps/'))
  else:
      raise BaseException('Sweep ID invalid!')
  sweep_id = exp_id
  wandb.login()
  wandb.agent(sweep_id, train_config, count=exp_n)


# __main__
################################################################################
if __name__ == '__main__':
    main()
