import sys
import dataset
import seqnn
import trainer
import basenji_model_for_model_zoo as model_zoo
import json
import os
import time
import glob
from scipy.stats import pearsonr
import shutil
import hashlib
import sys
import pandas as pd
from sklearn.metrics import r2_score
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


import glob
import json
import os
# import pdb
# import sys

from natsort import natsorted
import numpy as np
import tensorflow as tf

 ################################################################
 # functions for loading tfr files into tfr dataset
 ################################################################

def load_stats(data_dir):
  data_stats_file = '%s/statistics.json' % data_dir
  with open(data_stats_file) as data_stats_open:
      data_stats = json.load(data_stats_open)
  return data_stats

def batches_per_epoch(num_seqs, batch_size):
  return num_seqs // batch_size

def file_to_records(filename):
  return tf.data.TFRecordDataset(filename, compression_type='ZLIB')

def generate_parser(seq_length, target_length, num_targets):
  def parse_proto(example_protos):
    """Parse TFRecord protobuf."""
    # TFRecord constants
    TFR_INPUT = 'sequence'
    TFR_OUTPUT = 'target'

    # define features
    features = {
      TFR_INPUT: tf.io.FixedLenFeature([], tf.string),
      TFR_OUTPUT: tf.io.FixedLenFeature([], tf.string)
    }

    # parse example into features
    parsed_features = tf.io.parse_single_example(example_protos, features=features)

    # decode sequence
    sequence = tf.io.decode_raw(parsed_features[TFR_INPUT], tf.uint8)
    sequence = tf.reshape(sequence, [seq_length, 4])
    sequence = tf.cast(sequence, tf.float32)

    # decode targets
    targets = tf.io.decode_raw(parsed_features[TFR_OUTPUT], tf.float16)
    targets = tf.reshape(targets, [target_length, num_targets])
    targets = tf.cast(targets, tf.float32)

    return sequence, targets

  return parse_proto



def make_dataset(data_dir, split_label, data_stats, batch_size=64, seed=None):
    seq_length = data_stats['seq_length']
    target_length = data_stats['target_length']
    num_targets = data_stats['num_targets']
    tfr_path = '%s/tfrecords/%s-*.tfr' % (data_dir, split_label)
    num_seqs = data_stats['%s_seqs' % split_label]
    tfr_files = natsorted(glob.glob(tfr_path))
    dataset = tf.data.Dataset.list_files(tf.constant(tfr_files), shuffle=False)
    dataset = dataset.flat_map(file_to_records)
    dataset = dataset.map(generate_parser(seq_length, target_length, num_targets))
    if seed:
        dataset = dataset.shuffle(32, seed=seed)
    else:
        dataset = dataset.shuffle(32)
    # dataset = dataset.batch(64)
    # batch
    dataset = dataset.batch(batch_size)

    # prefetch
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

def tfr_to_np(data, choose, array_shape):
    if choose=='x':
        data_part = data.map(lambda x,y: x)
    elif choose=='y':
        data_part = data.map(lambda x,y: y)
    data_np = np.zeros(array_shape)
    # load data to a numpy array
    iter_data = iter(data_part)
    j=0
    for i in iter_data:
        n_seqs = i.shape[0]
        data_np[j:j+n_seqs,:,:] = i
        j+=n_seqs
    return data_np


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
