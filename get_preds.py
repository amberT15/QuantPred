import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px
import wandb
import tensorflow as tf
import h5py, os
import numpy as np
import plotly.express as px
import flask
import scipy.stats
import plotly.graph_objects as go
from modelzoo import GELU
import h5py
import custom_fit
import util
# from metrics import *
import sys


# y axis fixed somewhere in the Average
# remove mse
# dropdown menu for cell lines
# plot extra cell lines



fold_name='test'
N=10

#get data and model
CELL_LINE = int(sys.argv[1])

run_path = sys.argv[2] #'/home/shush/profile/QuantPred/wandb/run-20210623_073933-vwq5gdk5/files/'
data_dir = 'datasets/chr8/complete/random_chop/i_2048_w_1/'
h5_path = os.path.join(run_path, str(CELL_LINE) + '_dash.h5')

if not os.path.isfile(h5_path):
    model_path = os.path.join(run_path, 'best_model.h5')
    trained_model = tf.keras.models.load_model(model_path, custom_objects={"GELU": GELU})
    trained_model.compile(optimizer="Adam", loss='mse', metrics=['mse'])
    foldset = util.make_dataset(data_dir, fold_name, util.load_stats(data_dir), coords=True, shuffle=True)

    #get bin size
    n_bins = trained_model.layers[-1].output_shape[1]
    l_seq = trained_model.layers[0].input_shape[0][1]
    BIN_SIZE = l_seq // n_bins

    x_list = []
    y_list = []
    y_ori = []
    c_list = []
    pred_list = []
    for i, (coord, x, y) in enumerate(foldset):
        if i < N:
            y_ori.append(y)
            x_cropped, y_cropped = custom_fit.center_crop(x,y,2048)
            y_binned = custom_fit.bin_resolution(y_cropped, BIN_SIZE)
            c_list += [str(c).strip('b\'chr').strip('\'') for c in coord.numpy()]
            x_list.append(x_cropped)
            y_list.append(y_binned[:,:,CELL_LINE])
            pred_list.append(trained_model.predict(x_cropped)[:,:,CELL_LINE])
        else:
            break



    h = h5py.File(h5_path, 'w')
    h.create_dataset('y_true', data= np.concatenate(y_list))
    h.create_dataset('y_pred', data= np.concatenate(pred_list))
    h.create_dataset('bin_size', data= np.array(BIN_SIZE))
    h.close()
