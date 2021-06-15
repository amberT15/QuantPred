import sys
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
from natsort import natsorted
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

def generate_parser(seq_length, target_length, num_targets, coords):
  def parse_proto(example_protos):
    """Parse TFRecord protobuf."""
    # TFRecord constants
    TFR_COORD = 'coordinate'
    TFR_INPUT = 'sequence'
    TFR_OUTPUT = 'target'

    # define features
    features = {
      TFR_COORD: tf.io.FixedLenFeature([], tf.string),
      TFR_INPUT: tf.io.FixedLenFeature([], tf.string),
      TFR_OUTPUT: tf.io.FixedLenFeature([], tf.string)
    }

    # parse example into features
    parsed_features = tf.io.parse_single_example(example_protos, features=features)

    # decode coords
    coordinate = parsed_features[TFR_COORD]

    # decode sequence
    sequence = tf.io.decode_raw(parsed_features[TFR_INPUT], tf.uint8)
    sequence = tf.reshape(sequence, [seq_length, 4])
    sequence = tf.cast(sequence, tf.float32)

    # decode targets
    targets = tf.io.decode_raw(parsed_features[TFR_OUTPUT], tf.float16)
    targets = tf.reshape(targets, [target_length, num_targets])
    targets = tf.cast(targets, tf.float32)
    if coords:
        return coordinate, sequence, targets
    else:
        return sequence, targets

  return parse_proto



def make_dataset(data_dir, split_label, data_stats, batch_size=64, seed=None, coords=False):
    seq_length = data_stats['seq_length']
    target_length = data_stats['target_length']
    num_targets = data_stats['num_targets']
    tfr_path = '%s/tfrecords/%s-*.tfr' % (data_dir, split_label)
    num_seqs = data_stats['%s_seqs' % split_label]
    tfr_files = natsorted(glob.glob(tfr_path))
    dataset = tf.data.Dataset.list_files(tf.constant(tfr_files), shuffle=False)

    # train
    # if split_label == 'train':
    if (split_label == 'train'):
      # repeat
      #dataset = dataset.repeat()

      # interleave files
      dataset = dataset.interleave(map_func=file_to_records,
        cycle_length=4,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

      # shuffle
      dataset = dataset.shuffle(buffer_size=32,
        reshuffle_each_iteration=True)

    # valid/test
    else:
      # flat mix files
      dataset = dataset.flat_map(file_to_records)

    dataset = dataset.map(generate_parser(seq_length, target_length, num_targets, coords))
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
    for i in array_shape[0]:
        n_seqs = i.shape[0]
        data_np[j:j+n_seqs,:,:] = i
        j+=n_seqs
    return data_np
