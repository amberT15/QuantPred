import numpy as np
import pandas as pd
from pathlib import Path
import modelzoo
import tensorflow as tf
import yaml
import h5py
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns
import pandas as pd
import scipy
import sklearn.metrics as skm


def metrify(func):
    def wrapper(y_true,y_pred, metric=False):
        if metric:
            y_true = tf.expand_dims(y_true, axis=-1)
            y_pred = tf.expand_dims(y_pred, axis=-1)
        return func(y_true,y_pred)
    return wrapper



def pearsonr_per_seq(y, pred):
    y_true = tf.cast(y, 'float32')
    y_pred = tf.cast(pred, 'float32')

    product = tf.reduce_sum(tf.multiply(y_true, y_pred), axis=[1])

    true_sum = tf.reduce_sum(y_true, axis=[1])
    true_sumsq = tf.reduce_sum(tf.math.square(y_true), axis=[1])
    pred_sum = tf.reduce_sum(y_pred, axis=[1])
    pred_sumsq = tf.reduce_sum(tf.math.square(y_pred), axis=[1])

    count = tf.ones_like(y_true)
    count = tf.reduce_sum(count, axis=[1])

    true_mean = tf.divide(true_sum, count)
    true_mean2 = tf.math.square(true_mean)
    pred_mean = tf.divide(pred_sum, count)
    pred_mean2 = tf.math.square(pred_mean)

    term1 = product
    term2 = -tf.multiply(true_mean, pred_sum)
    term3 = -tf.multiply(pred_mean, true_sum)
    term4 = tf.multiply(count, tf.multiply(true_mean, pred_mean))
    covariance = term1 + term2 + term3 + term4

    true_var = true_sumsq - tf.multiply(count, true_mean2)
    pred_var = pred_sumsq - tf.multiply(count, pred_mean2)
    tp_var = tf.multiply(tf.math.sqrt(true_var), tf.math.sqrt(pred_var))
    correlation = tf.divide(covariance, tp_var)

    return tf.reduce_mean(correlation)


def calculate_pearsonr(target,pred):
    pearson_profile =np.zeros((target.shape[2],len(target)))
    
    for task_i in range(0,target.shape[2]):
        for sample_i in range(0,len(target)):
            pearson_profile[task_i,sample_i]=(pearsonr(target[sample_i][:,task_i],pred[sample_i][:,task_i])[0])
 
    return pearson_profile


def pearson_volin(pearson_profile,tasks,figsize=(20,5)):
    pd_dict = {}
    for i in range(0,len(tasks)):
        pd_dict[tasks[i]]=pearson_profile[i]

    pearsonr_pd = pd.DataFrame.from_dict(pd_dict)
    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.violinplot(data=pearsonr_pd)
    return fig
 

def pearson_box(pearson_profile,tasks,figsize=(20,5)):
    pd_dict = {}
    for i in range(0,len(tasks)):
        pd_dict[tasks[i]]=pearson_profile[i]
        
    pearsonr_pd = pd.DataFrame.from_dict(pd_dict)
    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.boxplot(data=pearsonr_pd)
    return fig

def permute_array(arr, axis=0):
    """Permute array along a certain axis

    Args:
      arr: numpy array
      axis: axis along which to permute the array
    """
    if axis == 0:
        return np.random.permutation(arr)
    else:
        return np.random.permutation(arr.swapaxes(0, axis)).swapaxes(0, axis)



def bin_counts_amb(x, binsize=2):
    """Bin the counts
    """
    if binsize == 1:
        return x
    assert len(x.shape) == 3
    outlen = x.shape[1] // binsize
    xout = np.zeros((x.shape[0], outlen, x.shape[2])).astype(float)
    for i in range(outlen):
        iterval = x[:, (binsize * i):(binsize * (i + 1)), :]
        has_amb = np.any(iterval == -1, axis=1)
        has_peak = np.any(iterval == 1, axis=1)
        # if no peak and has_amb -> -1
        # if no peak and no has_amb -> 0
        # if peak -> 1
        xout[:, i, :] = (has_peak - (1 - has_peak) * has_amb).astype(float)
    return xout

def auprc(y_true, y_pred):
    """Area under the precision-recall curve
    """
    y_true, y_pred = _mask_value_nan(y_true, y_pred)
    return skm.average_precision_score(y_true, y_pred)

def bin_counts_max(x, binsize=2):
    """Bin the counts
    """
    if binsize == 1:
        return x
    assert len(x.shape) == 3
    outlen = x.shape[1] // binsize
    xout = np.zeros((x.shape[0], outlen, x.shape[2]))
    for i in range(outlen):
        xout[:, i, :] = x[:, (binsize * i):(binsize * (i + 1)), :].max(1)
    return xout


MASK_VALUE = -1
def _mask_value_nan(y_true, y_pred, mask=MASK_VALUE):
    y_true, y_pred = _mask_nan(y_true, y_pred)
    return _mask_value(y_true, y_pred, mask)



def _mask_nan(y_true, y_pred):
    mask_array = ~np.isnan(y_true)
    if np.any(np.isnan(y_pred)):
        print("WARNING: y_pred contains {0}/{1} np.nan values. removing them...".
              format(np.sum(np.isnan(y_pred)), y_pred.size))
        mask_array = np.logical_and(mask_array, ~np.isnan(y_pred))
    return y_true[mask_array], y_pred[mask_array]


def _mask_value(y_true, y_pred, mask=MASK_VALUE):
    mask_array = y_true != mask
    return y_true[mask_array], y_pred[mask_array]

def eval_profile(yt, yp,
                 pos_min_threshold=0.05,
                 neg_max_threshold=0.01,
                 required_min_pos_counts=2.5,
                 binsizes=[1, 2, 4, 10]):
    """
    Evaluate the profile in terms of auPR

    Args:
      yt: true profile (counts)
      yp: predicted profile (fractions)
      pos_min_threshold: fraction threshold above which the position is
         considered to be a positive
      neg_max_threshold: fraction threshold bellow which the position is
         considered to be a negative
      required_min_pos_counts: smallest number of reads the peak should be
         supported by. All regions where 0.05 of the total reads would be
         less than required_min_pos_counts are excluded
    """
    # The filtering
    # criterion assures that each position in the positive class is
    # supported by at least required_min_pos_counts  of reads
    do_eval = yt.sum(axis=1).mean(axis=1) > required_min_pos_counts / pos_min_threshold

    # make sure everything sums to one
    yp = yp / yp.sum(axis=1, keepdims=True)
    fracs = yt / yt.sum(axis=1, keepdims=True)

    yp_random = permute_array(permute_array(yp[do_eval], axis=1), axis=0)
    out = []
    for binsize in binsizes:
        is_peak = (fracs >= pos_min_threshold).astype(float)
        ambigous = (fracs < pos_min_threshold) & (fracs >= neg_max_threshold)
        is_peak[ambigous] = -1
        y_true = np.ravel(bin_counts_amb(is_peak[do_eval], binsize))

        imbalance = np.sum(y_true == 1) / np.sum(y_true >= 0)
        n_positives = np.sum(y_true == 1)
        n_ambigous = np.sum(y_true == -1)
        frac_ambigous = n_ambigous / y_true.size

        # TODO - I used to have bin_counts_max over here instead of bin_counts_sum
        try:
            res = auprc(y_true,
                        np.ravel(bin_counts_max(yp[do_eval], binsize)))
            res_random = auprc(y_true,
                               np.ravel(bin_counts_max(yp_random, binsize)))
        except Exception:
            print('Exception Encountered')
            res = np.nan
            res_random = np.nan

        out.append({"binsize": binsize,
                    "auprc": res,
                    "random_auprc": res_random,
                    "n_positives": n_positives,
                    "frac_ambigous": frac_ambigous,
                    "imbalance": imbalance
                    })

    return pd.DataFrame.from_dict(out)
