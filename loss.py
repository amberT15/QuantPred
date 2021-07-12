import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

class zero_infl_poiss(tf.keras.losses.Loss):
    def __init__(self, name="zero_infl_poiss"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        tfd = tfp.distributions
        #multinomial part of loss function
        rate = tf.math.exp(y_pred)
        dim =y_true.shape
        nonzero_prob = tf.math.divide(tf.cast(
                                     tf.math.count_nonzero(y_pred, axis=(1, 2)), tf.float32),
                                     tf.cast(tf.size(y_pred), tf.float32))
        nonzero_prob = tf.expand_dims(nonzero_prob,axis =0)
        nonzero_prob=tf.tile(tf.transpose(nonzero_prob),[1,int(dim[1]*dim[2])])
        nonzero_prob = tf.reshape(nonzero_prob,dim)
        cat = tfd.Categorical(probs=tf.stack([1-nonzero_prob, nonzero_prob], -1))
        components = [tfd.Deterministic(loc=tf.zeros_like(rate)), tfd.Poisson(rate)]
        zip_dist = tfd.Mixture(cat=cat, components=components)
        # get the sequence length for normalization
        seqlen = tf.cast(dim[0],dtype=tf.float32)
        zip_loss = -tf.reduce_sum(zip_dist.log_prob(y_true)) / seqlen
        return zip_loss

class pearsonr_mse(tf.keras.losses.Loss):
    def __init__(self, name="pearsonr_mse", alpha=0.1):
        super().__init__(name=name)
        self.alpha=alpha
    def call(self, y_true, y_pred):
        #multinomial part of loss function
        pr_loss = basenjipearsonr()
        mse_loss = mse()
        mse_raw = mse_loss(y_true, y_pred)

        #sum with weight
        total_loss = pr_loss(y_true, y_pred) + self.alpha*mse_raw

        return total_loss


class fft_mse(tf.keras.losses.Loss):
    def __init__(self, name="fftmse"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):

        y_true = tf.cast(y_true, 'complex64')
        y_pred = tf.cast(y_pred, 'complex64')
        N, L, T = y_true.shape
        fft_y = tf.math.abs(tf.slice(tf.signal.fft(y_true),[0,0,0], [N, L//2, T]))
        fft_pred = tf.math.abs(tf.slice(tf.signal.fft(y_pred),[0,0,0], [N, L//2, T]))
        fft_diff = tf.math.subtract(fft_y, fft_pred)
        fft_diff = tf.dtypes.cast(fft_diff, 'float32')
        return tf.math.square(fft_diff)


class poisson(tf.keras.losses.Loss):
    def __init__(self, name="poisson"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        return tf.keras.losses.poisson(y_true, y_pred)

class log_poisson(tf.keras.losses.Loss):
    def __init__(self, name="log_poisson"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        y_true = tf.math.log(y_true+1)
        y_pred = tf.math.log(y_pred+1)
        return tf.keras.losses.poisson(y_true, y_pred)

class fftabs(tf.keras.losses.Loss):
    def __init__(self, name="fftabs"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, 'complex64')
        y_pred = tf.cast(y_pred, 'complex64')

        fft_y = tf.signal.fft(y_true)
        fft_pred = tf.signal.fft(y_pred)
        fft_diff = tf.math.subtract(fft_y, fft_pred)
        return tf.math.abs(fft_diff)

class mse(tf.keras.losses.Loss):
    def __init__(self, name="mse"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        return tf.keras.losses.MSE(y_true,y_pred)

class multinomialnll(tf.keras.losses.Loss):
    def __init__(self, name="multinomial"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        logits_perm = tf.transpose(y_pred, (0, 2, 1))
        true_counts_perm = tf.transpose(y_true, (0, 2, 1))
        counts_per_example = tf.reduce_sum(true_counts_perm, axis=-1)
        dist = tfp.distributions.Multinomial(total_count=counts_per_example,
                                                logits=logits_perm)
        # get the sequence length for normalization
        seqlen = tf.cast(tf.shape(y_true)[0],dtype=tf.float32)
        return -tf.reduce_sum(dist.log_prob(true_counts_perm)) / seqlen

class multinomialnll_mse(tf.keras.losses.Loss):
    def __init__(self, name="multinomial_mse", alpha=10):
        super().__init__(name=name)
        self.alpha=alpha
    def call(self, y_true, y_pred):
        #multinomial part of loss function
        logits_perm = tf.transpose(y_pred[0], (0, 2, 1))
        true_counts_perm = tf.transpose(y_true[0], (0, 2, 1))
        counts_per_example = tf.reduce_sum(true_counts_perm, axis=-1)
        dist = tfp.distributions.Multinomial(total_count=counts_per_example,
                                                logits=logits_perm)
        # get the sequence length for normalization
        seqlen = tf.cast(tf.shape(y_true[0])[0],dtype=tf.float32)
        mult_loss = -tf.reduce_sum(dist.log_prob(true_counts_perm)) / seqlen

        #MSE part of loss function
        mse_loss = tf.keras.losses.MSE(y_true[1], y_pred[1])

        #sum with weight
        total_loss = mult_loss + self.alpha*mse_loss

        return total_loss



class log_multinomialnll_mse(tf.keras.losses.Loss):
    def __init__(self, name="multinomial"):
        super().__init__(name=name)

    def call(self, y_true, y_pred, true_cov, pred_cov):
        #do log on all inputs
        y_true = tf.math.log(y_true)
        y_pred = tf.math.log(y_pred)
        true_cov = tf.math.log(true_cov)
        pred_cov = tf.math.log(pred_cov)

        #multinomial part of loss function
        logits_perm = tf.transpose(y_pred, (0, 2, 1))
        true_counts_perm = tf.transpose(y_true, (0, 2, 1))
        counts_per_example = tf.reduce_sum(true_counts_perm, axis=-1)
        dist = tfp.distributions.Multinomial(total_count=counts_per_example,
                                                logits=logits_perm)
        # get the sequence length for normalization
        seqlen = tf.cast(tf.shape(y_true)[0],dtype=tf.float32)
        mult_loss = -tf.reduce_sum(dist.log_prob(true_counts_perm)) / seqlen

        #MSE part of loss function
        mse_loss = tf.keras.losses.MSE(true_cov, pred_cov)

        #sum with weight
        total_loss = mult_loss + 10*mse_loss

        return total_loss

class basenjipearsonr (tf.keras.losses.Loss):
    def __init__(self, name="basenjipearsonr"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, 'float32')
        y_pred = tf.cast(y_pred, 'float32')

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


class r2 (tf.keras.losses.Loss):
    def __init__(self, name="r2"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, 'float32')
        y_pred = tf.cast(y_pred, 'float32')
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



def logthis(func):
    def wrapper(y_true,y_pred, metric=False):
        if metric:
            y_true = tf.math.log(y_true)
            y_pred = tf.math.log(y_pred)
        return func(y_true,y_pred)
    return wrapper
