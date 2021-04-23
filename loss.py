import tensorflow as tf

def poisson(y_true,y_pred):
    return tf.keras.losses.poisson(y_true, y_pred)

def multinomial_nll(y_true,y_pred):
    logits_perm = tf.transpose(logits, (0, 2, 1))
    true_counts_perm = tf.transpose(true_counts, (0, 2, 1))

    counts_per_example = tf.reduce_sum(true_counts_perm, axis=-1)

    dist = tfp.distributions.Multinomial(total_count=counts_per_example,
                                                logits=logits_perm)

    # get the sequence length for normalization
    seqlen = tf.cast(tf.shape(true_counts)[0],dtype=tf.float32)

    return -tf.reduce_sum(dist.log_prob(true_counts_perm)) / seqlen

def mse(y_true,y_pred):
    return tf.kera.slosses.MSE(y_true,y_pred)
