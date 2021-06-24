#!/usr/bin/env python
"""SeqNN trainer"""
import time
from packaging import version
import pdb
from wandb.keras import WandbCallback
from wandb_callbacks import *
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

import metrics



class Trainer:
  def __init__(self, train_data, eval_data, out_dir, train_epochs_max, loss,
               strategy=None, num_gpu=1, keras_fit=True, learning_rate=0.01,
                patience=10, clip_norm_default=2, optimizer='sgd', momentum=0.99):
    # self.params = params
    self.train_data = train_data
    if type(self.train_data) is not list:
      self.train_data = [self.train_data]
    self.eval_data = eval_data
    if type(self.eval_data) is not list:
      self.eval_data = [self.eval_data]
    self.model_path = out_dir
    self.strategy = strategy
    self.num_gpu = num_gpu
    self.learning_rate = learning_rate
    self.optimizer = optimizer
    self.clip_norm_default = clip_norm_default
    self.momentum = momentum
    self.batch_size = self.train_data[0].batch_size
    self.compiled = False
    self.loss_fn = loss


    # early stopping
    # self.patience = self.params.get('patience', 20)
    self.patience = patience

    # compute batches/epoch
    self.train_epoch_batches = [td.batches_per_epoch() for td in self.train_data]
    self.eval_epoch_batches = [ed.batches_per_epoch() for ed in self.eval_data]
    self.train_epochs_min = 1
    self.train_epochs_max = int(train_epochs_max)

    # dataset
    self.num_datasets = len(self.train_data)
    self.dataset_indexes = []
    for di in range(self.num_datasets):
      self.dataset_indexes += [di]*self.train_epoch_batches[di]
    self.dataset_indexes = np.array(self.dataset_indexes)

    # loss

    # self.spec_weight = 1
    # self.loss = 'poisson'
    # self.loss_fn = parse_loss(self.loss, self.strategy, keras_fit, self.spec_weight)

    # optimizer
    self.make_optimizer()

  def compile(self, seqnn_model):
    for model in [seqnn_model]:
      # if self.loss == 'bce':
      #   model_metrics = [metrics.SeqAUC(curve='ROC'), metrics.SeqAUC(curve='PR')]
      # else:
      num_targets = model.output_shape[-1]
      print('num_targets ', num_targets)

      # model_metrics = [metrics.PearsonR(num_targets)]

      # model.compile(loss=self.loss_fn,
      #               optimizer=self.optimizer,
      #               metrics=model_metrics)


      model.compile(loss=self.loss_fn,
                    optimizer=self.optimizer)
    self.compiled = True

  def fit_keras(self, seqnn_model, callbacks=[]):
    if not self.compiled:
      self.compile(seqnn_model)

    # if self.loss == 'bce':
    #   early_stop = EarlyStoppingMin(monitor='val_loss', mode='min', verbose=1,
    #                    patience=self.patience, min_epoch=self.train_epochs_min)
    #   save_best = tf.keras.callbacks.ModelCheckpoint('%s/model_best.h5'%self.out_dir,
    #                                                  save_best_only=True, mode='min',
    #                                                  monitor='val_loss', verbose=1)
    # else:
    # wandb_callback = WandbCallback()
    # save_callback = ModelCheckpoint(os.path.join(wandb.run.dir, "best_model.tf"), save_weights_only=True, save_best_only=True)
    # if no callbacks set to default
    if len(callbacks)==0:
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                             factor=0.2, # TODO .2
                                                             patience=3, # 3
                                                             min_lr=1e-7,
                                                             mode='min',
                                                             verbose=1)
        early_stop = EarlyStoppingMin(monitor='val_pearsonr', mode='max', verbose=1,
                           patience=self.patience, min_epoch=self.train_epochs_min)
        save_best = tf.keras.callbacks.ModelCheckpoint(self.model_path,
                                                         save_best_only=True, mode='max',
                                                         monitor='val_pearsonr', verbose=1)

        callbacks = [early_stop, reduce_lr, save_best]


    history = seqnn_model.fit(
      self.train_data[0].dataset,
      epochs=self.train_epochs_max,
      steps_per_epoch=self.train_epoch_batches[0],
      callbacks=callbacks,
      validation_data=self.eval_data[0].dataset,
      validation_steps=self.eval_epoch_batches[0])
    self.history = history
    return self.history




  def make_optimizer(self):
    # schedule (currently OFF)
    initial_learning_rate = self.learning_rate

    if version.parse(tf.__version__) < version.parse('2.2'):
      clip_norm_default = 1000000
    else:
      clip_norm_default = None

    clip_norm = clip_norm_default
    # optimizer
    # optimizer_type = optimizer
    if self.optimizer == 'adam':

      self.optimizer = tf.keras.optimizers.Adam(
          learning_rate=self.learning_rate)

    elif self.optimizer in ['sgd', 'momentum']:
      self.optimizer = tf.keras.optimizers.SGD(
          learning_rate=self.learning_rate,
          momentum=0.99,
          clipnorm=clip_norm)

    else:
      print('Cannot recognize optimization algorithm %s' % self.optimizer)
      exit(1)

class EarlyStoppingMin(tf.keras.callbacks.EarlyStopping):
  """Stop training when a monitored quantity has stopped improving.
  Arguments:
      min_epoch: Minimum number of epochs before considering stopping.

  """
  def __init__(self, min_epoch=0, **kwargs):
    super(EarlyStoppingMin, self).__init__(**kwargs)
    self.min_epoch = min_epoch

  def on_epoch_end(self, epoch, logs=None):
    current = self.get_monitor_value(logs)
    if current is None:
      return
    if self.monitor_op(current - self.min_delta, self.best):
      self.best = current
      self.wait = 0
      if self.restore_best_weights:
        self.best_weights = self.model.get_weights()
    else:
      self.wait += 1
      if epoch >= self.min_epoch and self.wait >= self.patience:
        self.stopped_epoch = epoch
        self.model.stop_training = True
        if self.restore_best_weights:
          if self.verbose > 0:
            print('Restoring model weights from the end of the best epoch.')
          self.model.set_weights(self.best_weights)

class Cyclical1LearningRate(tf.keras.optimizers.schedules.LearningRateSchedule):
  """A LearningRateSchedule that uses cyclical schedule.
  https://yashuseth.blog/2018/11/26/hyper-parameter-tuning-best-practices-learning-rate-batch-size-momentum-weight-decay/
  """

  def __init__(
    self,
    initial_learning_rate,
    maximal_learning_rate,
    final_learning_rate,
    step_size,
    name: str = "Cyclical1LearningRate",
  ):
    super().__init__()
    self.initial_learning_rate = initial_learning_rate
    self.maximal_learning_rate = maximal_learning_rate
    self.final_learning_rate = final_learning_rate
    self.step_size = step_size
    self.name = name

  def __call__(self, step):
    with tf.name_scope(self.name or "Cyclical1LearningRate"):
      initial_learning_rate = tf.convert_to_tensor(self.initial_learning_rate,
        name="initial_learning_rate")
      dtype = initial_learning_rate.dtype
      maximal_learning_rate = tf.cast(self.maximal_learning_rate, dtype)
      final_learning_rate = tf.cast(self.final_learning_rate, dtype)

      step_size = tf.cast(self.step_size, dtype)
      cycle = tf.floor(1 + step / (2 * step_size))
      x = tf.abs(step / step_size - 2 * cycle + 1)

      lr = tf.where(step > 2*step_size,
                    final_learning_rate,
                    initial_learning_rate + (
                      maximal_learning_rate - initial_learning_rate
                      ) * tf.maximum(tf.cast(0, dtype), (1 - x)))
      return lr

  def get_config(self):
      return {
          "initial_learning_rate": self.initial_learning_rate,
          "maximal_learning_rate": self.maximal_learning_rate,
          "final_learning_rate": self.final_learning_rate,
          "step_size": self.step_size,
      }
