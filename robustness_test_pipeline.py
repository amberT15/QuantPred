import tensorflow as tf
import h5py
import explain
import custom_fit
import modelzoo
from loss import *
import os,json
import util
import pandas as pd
import importlib
import seaborn as sns
import scipy
import matplotlib.pyplot as plt
import metrics
import csv
import sys
import h5py

test_data = './datasets/VCF/filtered_test.h5'
f =  h5py.File(test_data, "r")
x = f['x_test'][()]
y = f['y_test'][()]
f.close()

model_path = sys.argv[1]
id = model_path.split('-')[-1]


saliency_file = h5py.File('./datasets/VCF/saliency_eval_'+id+'.h5','w')

pred_file = h5py.File('./datasets/VCF/pred_eval_'+id+'.h5','w')

model = modelzoo.load_model(model_path,compile=True)
var_saliency,var_pred = explain.batch_robustness_test(x,y,
                                                  model,visualize = False,shift_num =10)

saliency_file.create_dataset('variance',data = var_saliency)
pred_file.create_dataset('variance',data = var_pred)
saliency_file.close()
pred_file.close()


tf.keras.backend.clear_session()
