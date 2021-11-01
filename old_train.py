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


def main():
    usage = "usage: %prog [options] <data_dir> <model name> <loss type>"
    parser = OptionParser(usage)
    parser.add_option(
        "-o",
        dest="out_dir",
        default=".",
        help="Output where model and pred will be saved",
    )
    parser.add_option(
        "-e",
        dest="n_epochs",
        default=100,
        type="int",
        help="N epochs [Default: %default]",
    )
    parser.add_option(
        "-p",
        dest="earlystop_p",
        default=10,
        type="int",
        help="Early stopping patience [Default: %default]",
    )
    parser.add_option(
        "-l",
        dest="l_rate",
        default=0.001,
        type="float",
        help="learning rate [Default: %default]",
    )
    (options, args) = parser.parse_args()

    if len(args) != 3:
        parser.error("Must provide data dir, model architecture name, and loss")
    else:
        data_dir = args[0]
        model_name_str = args[1]
        loss_type_str = args[2]

    model_name = eval(model_name_str)
    loss_type = eval(loss_type_str)
    json_path = os.path.join(data_dir, "statistics.json")
    batch_size = 64
    # load data
    train_data = util.make_dataset(data_dir, "train", util.load_stats(data_dir))
    valid_data = util.make_dataset(data_dir, "valid", util.load_stats(data_dir))
    test_data = util.make_dataset(data_dir, "test", util.load_stats(data_dir))

    # create output folder if not present
    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)
    # load parameters of the dataset
    with open(json_path) as json_file:
        params = json.load(json_file)
    input_size = params["seq_length"]
    num_targets = params["num_targets"]
    n_seqs = params["valid_seqs"]
    output_length = int(params["seq_length"] / params["pool_width"])
    # precompute batch sizes
    train_n_batches = util.batches_per_epoch(params["train_seqs"], batch_size)
    valid_n_batches = util.batches_per_epoch(params["valid_seqs"], batch_size)
    print("Input size is {}, number of TFs is {}".format(input_size, num_targets))
    data_folder = os.path.basename(os.path.normpath(data_dir))
    prefix = "{}_{}_{}".format(data_folder, model_name_str, loss_type_str)
    print("Saving outputs using prefix " + prefix)
    out_model_path = os.path.join(options.out_dir, "model_" + prefix + ".h5")
    out_pred_path = os.path.join(options.out_dir, "pred_" + prefix + ".h5")

    # get the model architecture from the model zoo
    # if model_name=='basenji_small':
    #     model = modelzoo.basenji_small((input_size, 4), num_targets)
    model = model_name((input_size, 4), (output_length, num_targets))

    model.compile(tf.keras.optimizers.Adam(lr=options.l_rate), loss=loss_type)

    earlystop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=options.earlystop_p,
        verbose=1,
        mode="min",
        baseline=None,
        restore_best_weights=True,
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        out_model_path, monitor="val_loss", verbose=1, save_best_only=True, mode="min"
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=4, min_lr=1e-7, mode="min", verbose=1
    )
    history = model.fit(
        train_data,
        epochs=options.n_epochs,
        steps_per_epoch=train_n_batches,
        callbacks=[earlystop, checkpoint, reduce_lr],
        validation_data=valid_data,
        validation_steps=valid_n_batches,
    )

    test_y = util.tfr_to_np(
        test_data, "y", (params["test_seqs"], output_length, params["num_targets"])
    )
    test_x = util.tfr_to_np(
        test_data, "x", (params["test_seqs"], params["seq_length"], 4)
    )
    test_pred = model.predict(test_x)
    hf = h5py.File(out_pred_path, "w")
    hf.create_dataset("test_x", data=test_x)
    hf.create_dataset("test_y", data=test_y)
    hf.create_dataset("test_pred", data=test_pred)
    hf.close()


# __main__
################################################################################
if __name__ == "__main__":
    main()
