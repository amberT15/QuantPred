import tensorflow as tf
import tensorflow_probability as tfp


def logthis(func):
    def wrapper(y_true,y_pred, metric=False):
        if metric:
            y_true = tf.math.log(y_true)
            y_pred = tf.math.log(y_pred)
        return func(y_true,y_pred)
    return wrapper

def fft_abs(y, pred):
  y_true = tf.cast(y, 'complex64')
  y_pred = tf.cast(pred, 'complex64')

  fft_y = tf.signal.fft(y_true)
  fft_pred = tf.signal.fft(y_pred)
  fft_diff = tf.math.subtract(fft_y, fft_pred)
  return tf.math.abs(fft_diff)

def fft_mse(y, pred):
  y_true = tf.cast(y, 'complex64')
  y_pred = tf.cast(pred, 'complex64')

  fft_y = tf.signal.fft(y_true)
  fft_pred = tf.signal.fft(y_pred)
  fft_diff = tf.math.subtract(fft_y, fft_pred)
  fft_diff = tf.dtypes.cast(fft_diff, 'float32')
  return tf.math.square(fft_diff)


def poisson(y_true,y_pred):
    return tf.keras.losses.poisson(y_true, y_pred)

def multinomial_nll(y_true,y_pred):
    logits_perm = tf.transpose(y_pred, (0, 2, 1))
    true_counts_perm = tf.transpose(y_true, (0, 2, 1))
    counts_per_example = tf.reduce_sum(true_counts_perm, axis=-1)
    dist = tfp.distributions.Multinomial(total_count=counts_per_example,
                                                logits=logits_perm)
    # get the sequence length for normalization
    seqlen = tf.cast(tf.shape(y_true)[0],dtype=tf.float32)
    return -tf.reduce_sum(dist.log_prob(true_counts_perm)) / seqlen

def mse(y_true,y_pred):
    return tf.keras.losses.MSE(y_true,y_pred)


def pearsonr(y, pred):
    y_true = tf.cast(y, 'float32')
    y_pred = tf.cast(pred, 'float32')

    product = tf.reduce_sum(tf.multiply(y_true, y_pred), axis=[0,1])
    true_sum = tf.reduce_sum(y_true, axis=[0,1])
    true_sumsq = tf.reduce_sum(tf.math.square(y_true), axis=[0,1])
    pred_sum = tf.reduce_sum(y_pred, axis=[0,1])
    pred_sumsq = tf.reduce_sum(tf.math.square(y_pred), axis=[0,1])
    count = tf.ones_like(y_true)
    count = tf.reduce_sum(count, axis=[0,1])
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


    return -tf.reduce_mean(correlation)

def r2(y, pred):

    y_true = tf.cast(y, 'float32')
    y_pred = tf.cast(pred, 'float32')
    shape = y_true.shape[-1]
    true_sum = tf.reduce_sum(y_true, axis=[0,1])
    true_sumsq = tf.reduce_sum(tf.math.square(y_true), axis=[0,1])
    product = tf.reduce_sum(tf.multiply(y_true, y_pred), axis=[0,1])
    pred_sumsq = tf.reduce_sum(tf.math.square(y_pred), axis=[0,1])
    count = tf.ones_like(y_true)
    count = tf.reduce_sum(count, axis=[0,1])

    true_mean = tf.divide(true_sum, count)
    true_mean2 = tf.math.square(true_mean)

    total = true_sumsq - tf.multiply(count, true_mean2)

    resid1 = pred_sumsq
    resid2 = -2*product
    resid3 = true_sumsq
    resid = resid1 + resid2 + resid3

    r2 = tf.ones_like(shape, dtype=tf.float32) - tf.divide(resid, total)
    return -tf.reduce_mean(r2)
