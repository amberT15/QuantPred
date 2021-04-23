import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp


class GELU(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(GELU, self).__init__(**kwargs)
    def call(self, x):
        # return tf.keras.activations.sigmoid(1.702 * x) * x
        return tf.keras.activations.sigmoid(tf.constant(1.702) * x) * x