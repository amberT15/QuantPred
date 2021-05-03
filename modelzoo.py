import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp


class GELU(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(GELU, self).__init__(**kwargs)
    def call(self, x):
        # return tf.keras.activations.sigmoid(1.702 * x) * x
        return tf.keras.activations.sigmoid(tf.constant(1.702) * x) * x
    
    
def bpnet(tasks,input_shape,strand_num = 1,):
    #body
    input = keras.layers.Input(shape=input_shape)
    x = keras.layers.Conv1D(64,kernel_size=25,padding ='same',activation = 'relu')(input)
    for i in range(1,10):
        conv_x = keras.layers.Conv1D(64,kernel_size = 3, padding = 'same', activation = 'relu', dilation_rate = 2**i)(x)
        x = keras.layers.Add()([conv_x,x])
    
    bottleneck = x

    #heads
    outputs = []
    for task in tasks:
        px = keras.layers.Reshape((-1,1,64))(bottleneck)
        px = keras.layers.Conv2DTranspose(strand_num,kernel_size = (25,1),padding = 'same')(px)
        px = keras.layers.Reshape((-1,strand_num),name=task+'/profile')(px)
        outputs.append(px)
    
    model = keras.models.Model([input],outputs)
    return model