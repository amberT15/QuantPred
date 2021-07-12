import numpy as np
import tensorflow as tf
from tensorflow import keras
import math
import matplotlib.pyplot as plt
import pandas as pd
import logomaker

def plot_saliency(saliency_map):

    fig, axs = plt.subplots(saliency_map.shape[0],1,figsize=(200,5*saliency_map.shape[0]))
    for n, w in enumerate(saliency_map):
        ax = axs[n]
        #plot saliency map representation
        saliency_df = pd.DataFrame(w.numpy(), columns = ['A','C','G','T'])
        logomaker.Logo(saliency_df, ax=ax)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('none')
        ax.xaxis.set_ticks_position('none')
        plt.xticks([])
        plt.yticks([])
    return plt


def select_top_pred(pred,num_task,top_num):

    task_top_list = []
    for i in range(0,num_task):
        task_profile = pred[:,:,i]
        task_mean =np.squeeze(np.mean(task_profile,axis = 1))
        task_index = task_mean.argsort()[-top_num:]
        task_top_list.append(task_index)
    task_top_list = np.array(task_top_list)
    return task_top_list

def vcf_test(alt,ref,model):
    pred_ref = model.predict(ref)
    pred_alt = model.predict(alt)
    ref_pred_cov = np.sum(ref_pred,axis = 1)
    alt_pred_cov = np.sum(alt_pred,axis = 1)

    cell_diff = alt_pred_cov-ref_pred_cov
    total_diff = np.sum(cell_diff, axis = 1)

    cell_fold = np.log(alt_pred_cov) - np.log(ref_pred_cov)
    total_fold = np.mean(cell_fold,axis = 1)

    return cell_diff,total_diff,cell_fold,total_fold

def complete_saliency(X,model,class_index,func = tf.math.reduce_mean):
  """fast function to generate saliency maps"""
  if not tf.is_tensor(X):
    X = tf.Variable(X)

  with tf.GradientTape() as tape:
    tape.watch(X)
    if class_index is not None:
      outputs = func(model(X)[:,:,class_index])
    else:
      raise ValueError('class index must be provided')
  return tape.gradient(outputs, X)


def peak_saliency_map(X, model, class_index,window_size,func=tf.math.reduce_mean):
    """fast function to generate saliency maps"""
    if not tf.is_tensor(X):
        X = tf.Variable(X)

    with tf.GradientTape() as tape:
        tape.watch(X)
        pred = model(X)

        peak_index = tf.math.argmax(pred[:,:,class_index],axis=1)
        batch_indices = []

        if int(window_size) > 50:
            bin_num = 1
        elif int(window_size) == 32:
            bin_num = 3
        else:
            bin_num = 50

        for i in range(0,X.shape[0]):
            column_indices = tf.range(peak_index[i]-int(bin_num/2),peak_index[i]+math.ceil(bin_num/2),dtype='int32')
            row_indices = tf.keras.backend.repeat_elements(tf.constant([i]),bin_num, axis=0)
            full_indices = tf.stack([row_indices, column_indices], axis=1)
            batch_indices.append([full_indices])
            outputs = func(tf.gather_nd(pred[:,:,class_index],batch_indices),axis=2)

        return tape.gradient(outputs, X)
