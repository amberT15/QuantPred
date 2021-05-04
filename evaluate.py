#!/usr/bin/env python

import numpy as np
import tensorflow as tf

# load pre-trained model
model_path = '/home/shush/profile/QuantPred/model_i_1024_w_32_basenji_small_poisson.h5'
loss_type = tf.keras.losses.Poisson()


custom_layers = {'GELU':modelzoo.GELU,
               'StochasticReverseComplement':modelzoo.StochasticReverseComplement,
               'StochasticShift':modelzoo.StochasticShift,
               'SwitchReverse':modelzoo.SwitchReverse}

model = tf.keras.models.load_model(model_path,
                           custom_objects=custom_layers,
                           compile=False)


model.compile(tf.keras.optimizers.Adam(lr=0.001), loss=loss_type)
