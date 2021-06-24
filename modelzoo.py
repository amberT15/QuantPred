import tensorflow as tf
import tensorflow.keras as keras
# import tensorflow_probability as tfp
import numpy as np


def basenjimod(input_shape, output_shape, add_dropout, filtN_1=64, filtN_2=64, filt_mlt=1.125,
               filtN_4=32, filtN_5=64, kern_1=15, kern_2=5, kern_3=5, kern_4=3,
               kern_5=1, filtN_list=None):

               # learning rate [0.001, 0.004, 0.0004] do not so for now set to 0.001,
               # dropout [0, 0.1, 0.2] add to grid, remove kern size
               # filtN_1 [64, 128] do not decrease (do not do 128 -> 64)
               # filtN_2 [64, 128, 256]
               # filtN_3 [64, 128, 256]
               # filtN_4 [64, 128, 256, 512, 1024]
    """
    Basenji model turned into a single function.
    inputs (None, seq_length, 4)
    """

    print('Using set of filter sizes for hyperparameter search')
    filt_drp_dict = {64: 0.1, 128: 0.2, 256: 0.3, 512: 0.4, 1024: 0.5}
    if filtN_list:
        print('Using set of filter sizes for hyperparameter search')
        filtN_1, filtN_2, filtN_4, filtN_5 = filtN_list
    else:
        filtN_list = filtN_1, filtN_2, filtN_4, filtN_5

    filt_drp_dict = {64: 0.1, 128: 0.2, 256: 0.3, 512: 0.4, 1024: 0.5}
    if add_dropout:
        drp1, drp2, drp4, drp5  = [filt_drp_dict[f] for f in filtN_list]
    else:
        drp1 = drp2 = drp4 = drp5 = 0

    # dict for choosing number of maxpools based on output shape
    layer_dict = {32: [1, False], # if bin size 32 add 1 maxpool and no maxpool of size 2
                  64: [1, True], # if bin size 64 add 1 maxpool and 1 maxpool of size 2
                  128: [2, False], # if bin size 128 add 2 maxpool and no maxpool of size 2
                  256: [2, True]} # if bin size 256 add 2 maxpool and 1 maxpool of size 2
    L, _ = input_shape
    n_bins, n_exp = output_shape
    l_bin = L // n_bins
    n_conv_tower, add_2max = layer_dict[l_bin]
#     n_conv_tower = np.log2(32)
    # print(l_bin, n_conv_tower, add_2max)
    sequence = tf.keras.Input(shape=input_shape, name='sequence')

    current = conv_block(sequence, filters=filtN_1, kernel_size=kern_1, activation='gelu', activation_end=None,
        strides=1, dilation_rate=1, l2_scale=0, dropout=drp1, conv_type='standard', residual=False,
        pool_size=8, batch_norm=True, bn_momentum=0.9, bn_gamma=None, bn_type='standard',
        kernel_initializer='he_normal', padding='same')

    current, rep_filters = conv_tower(current, filters_init=filtN_2, filters_mult=filt_mlt, repeat=n_conv_tower,
        pool_size=4, kernel_size=kern_2, dropout=drp2, batch_norm=True, bn_momentum=0.9,
        activation='gelu')

    if add_2max:
        filtN_3 = int(np.round(rep_filters*filt_mlt))
        if filtN_list:
            filt_drp_dict = {64: 0.1, 128: 0.2, 256: 0.3, 512: 0.4, 1024: 0.5}
            drp3 = filt_drp_dict[filtN_3]
        current = conv_block(current, filters=filtN_3, kernel_size=kern_3, activation='gelu', activation_end=None, #changed filter size 5
            strides=1, dilation_rate=1, l2_scale=0, dropout=drp3, conv_type='standard', residual=False,
            pool_size=2, batch_norm=True, bn_momentum=0.9, bn_gamma=None, bn_type='standard',
            kernel_initializer='he_normal', padding='same')

    current = dilated_residual(current, filters=filtN_4, kernel_size=kern_4, rate_mult=2,
        conv_type='standard', dropout=drp4, repeat=2, round=False, # repeat=4 TODO:figure out scaling factor for the number of repeats
        activation='gelu', batch_norm=True, bn_momentum=0.9)

    current = conv_block(current, filters=filtN_5, kernel_size=kern_5, activation='gelu',
        dropout=drp5, batch_norm=True, bn_momentum=0.9)

    outputs = dense_layer(current, n_exp, activation='softplus',
        batch_norm=True, bn_momentum=0.9)


    model = tf.keras.Model(inputs=sequence, outputs=outputs)

    return model


def basenji(input_shape, output_shape, augment_rc=True, augment_shift=3):
    """
    Basenji model turned into a single function.
    inputs (None, seq_length, 4)
    """
    n_exp = output_shape[-1]
    sequence = tf.keras.Input(shape=input_shape, name='sequence')
    #StochasticReverseComplement
    if augment_rc:
      current, reverse_bool = StochasticReverseComplement()(sequence)
    #StochasticShift
    if augment_shift != [0]:
      current = StochasticShift(augment_shift)(current)

    current = conv_block(current, filters=64, kernel_size=15, activation='gelu', activation_end=None,
        strides=1, dilation_rate=1, l2_scale=0, dropout=0, conv_type='standard', residual=False,
        pool_size=8, batch_norm=True, bn_momentum=0.9, bn_gamma=None, bn_type='standard',
        kernel_initializer='he_normal', padding='same')

    current, _ = conv_tower(current, filters_init=64, filters_mult=1.125, repeat=1,
        pool_size=4, kernel_size=5, batch_norm=True, bn_momentum=0.9,
        activation='gelu')

    current = dilated_residual(current, filters=32, kernel_size=3, rate_mult=2,
        conv_type='standard', dropout=0.25, repeat=2, round=False, # repeat=4
        activation='gelu', batch_norm=True, bn_momentum=0.9)

    current = conv_block(current, filters=64, kernel_size=1, activation='gelu',
        dropout=0.05, batch_norm=True, bn_momentum=0.9)

    current = dense_layer(current, n_exp, activation='softplus',
        batch_norm=True, bn_momentum=0.9)

    # switch reverse
    outputs = SwitchReverse()([current, reverse_bool])

    model = tf.keras.Model(inputs=sequence, outputs=outputs)
    # print(model.summary())
    return model

def mult_filt(n, factor):
    return int(np.round(n*factor))

def basenjiw1(input_shape, output_shape, augment_rc=True, augment_shift=3):
    """
    Basenji model turned into a single function.
    inputs (None, seq_length, 4)
    """
    n_exp = output_shape[-1]
    sequence = tf.keras.Input(shape=input_shape, name='sequence')
    current = tf.expand_dims(sequence, -2)

    current = conv_block(current, filters=64, kernel_size=15, activation='gelu', activation_end=None,
        strides=1, dilation_rate=1, l2_scale=0, dropout=0, conv_type='standard', residual=False,
        pool_size=8, batch_norm=True, bn_momentum=0.9, bn_gamma=None, bn_type='standard',
        kernel_initializer='he_normal', padding='same', w1=True)

    current, _ = conv_tower(current, filters_init=64, filters_mult=1.125, repeat=1,
        pool_size=4, kernel_size=5, batch_norm=True, bn_momentum=0.9,
        activation='gelu', w1=True)

    current = dilated_residual(current, filters=64, kernel_size=3, rate_mult=2,
        conv_type='standard', dropout=0.25, repeat=2, round=False, # repeat=4
        activation='gelu', batch_norm=True, bn_momentum=0.9, w1=True)

    current = conv_block(current, filters=64, kernel_size=1, activation='gelu',
        dropout=0.05, batch_norm=True, bn_momentum=0.9, w1=True)
    #TODO: task specific heads (in a new version)
    n_loops = input_shape[0] // current.shape[1]
    n_filters = 64
    for n in range(2):
      n_filters = mult_filt(n_filters, 1.125)
      print(n_filters)
      current = tf.keras.layers.Conv2DTranspose(
                filters=n_filters, kernel_size=(5,1), strides=(2,1), padding='same')(current)
      current = tf.keras.layers.UpSampling2D(size=(2,1))(current)

    current = tf.keras.layers.Conv2DTranspose(
          filters=mult_filt(n_filters, 1.125), kernel_size=(5,1), strides=(2,1), padding='same')(current)
    current = tf.keras.layers.Conv2D(mult_filt(n_filters, 1.125), 1)(current)
    current = dense_layer(current, n_exp, activation='softplus',
        batch_norm=True, bn_momentum=0.9)
    outputs = tf.squeeze(
      current, axis=2)
    # upsample

    model = tf.keras.Model(inputs=sequence, outputs=outputs)
    print(model.summary())
    return model



############################################################
# layers and helper functions
############################################################
def conv_block(inputs, filters=None, kernel_size=1, activation='relu', activation_end=None,
    strides=1, dilation_rate=1, l2_scale=0, dropout=0, conv_type='standard', residual=False,
    pool_size=1, batch_norm=False, bn_momentum=0.99, bn_gamma=None, bn_type='standard',
    kernel_initializer='he_normal', padding='same', w1=False):
  """Construct a single convolution block.

  Args:
    inputs:        [batch_size, seq_length, features] input sequence
    filters:       Conv1D filters
    kernel_size:   Conv1D kernel_size
    activation:    relu/gelu/etc
    strides:       Conv1D strides
    dilation_rate: Conv1D dilation rate
    l2_scale:      L2 regularization weight.
    dropout:       Dropout rate probability
    conv_type:     Conv1D layer type
    residual:      Residual connection boolean
    pool_size:     Max pool width
    batch_norm:    Apply batch normalization
    bn_momentum:   BatchNorm momentum
    bn_gamma:      BatchNorm gamma (defaults according to residual)

  Returns:
    [batch_size, seq_length, features] output sequence
  """

  # flow through variable current
  current = inputs

  # choose convolution type
  if conv_type == 'separable':
    conv_layer = tf.keras.layers.SeparableConv1D
  elif w1:
    conv_layer = tf.keras.layers.Conv2D
  else:
    conv_layer = tf.keras.layers.Conv1D

  if filters is None:
    filters = inputs.shape[-1]

  # activation
  current = activate(current, activation)

  # convolution
  current = conv_layer(
    filters=filters,
    kernel_size=kernel_size,
    strides=strides,
    padding='same',
    use_bias=False,
    dilation_rate=dilation_rate,
    kernel_initializer=kernel_initializer,
    kernel_regularizer=tf.keras.regularizers.l2(l2_scale))(current)

  # batch norm
  if batch_norm:
    if bn_gamma is None:
      bn_gamma = 'zeros' if residual else 'ones'
    if bn_type == 'sync':
      bn_layer = tf.keras.layers.experimental.SyncBatchNormalization
    else:
      bn_layer = tf.keras.layers.BatchNormalization
    current = bn_layer(
      momentum=bn_momentum,
      gamma_initializer=bn_gamma)(current)

  # dropout
  if dropout > 0:
    current = tf.keras.layers.Dropout(rate=dropout)(current)

  # residual add
  if residual:
    current = tf.keras.layers.Add()([inputs,current])

  # end activation
  if activation_end is not None:
    current = activate(current, activation_end)

  # Pool
  if pool_size > 1:
    if w1:
      current = tf.keras.layers.MaxPool2D(
        pool_size=pool_size,
        padding=padding)(current)
    else:
      current = tf.keras.layers.MaxPool1D(
        pool_size=pool_size,
        padding=padding)(current)

  return current

def conv_tower(inputs, filters_init, filters_mult=1, repeat=1, **kwargs):
  """Construct a reducing convolution block.

  Args:
    inputs:        [batch_size, seq_length, features] input sequence
    filters_init:  Initial Conv1D filters
    filters_mult:  Multiplier for Conv1D filters
    repeat:        Conv block repetitions

  Returns:
    [batch_size, seq_length, features] output sequence
  """

  # flow through variable current
  current = inputs

  # initialize filters
  rep_filters = filters_init

  for ri in range(repeat):
    # convolution
    current = conv_block(current,
      filters=int(np.round(rep_filters)),
      **kwargs)

    # update filters
    rep_filters *= filters_mult

  return current, rep_filters


def dilated_residual(inputs, filters, kernel_size=3, rate_mult=2,
    conv_type='standard', dropout=0, repeat=1, round=False, **kwargs):
  """Construct a residual dilated convolution block.
  """

  # flow through variable current
  current = inputs

  # initialize dilation rate
  dilation_rate = 1.0

  for ri in range(repeat):
    rep_input = current

    # dilate
    current = conv_block(current,
      filters=filters,
      kernel_size=kernel_size,
      dilation_rate=int(np.round(dilation_rate)),
      conv_type=conv_type,
      bn_gamma='ones',
      **kwargs)

    # return
    current = conv_block(current,
      filters=rep_input.shape[-1],
      dropout=dropout,
      bn_gamma='zeros',
      **kwargs)

    # residual add
    current = tf.keras.layers.Add()([rep_input,current])

    # update dilation rate
    dilation_rate *= rate_mult
    if round:
      dilation_rate = np.round(dilation_rate)

  return current


# depracated, poorly named
def dense_layer(inputs, units, activation='linear', kernel_initializer='he_normal',
          l2_scale=0, l1_scale=0, **kwargs):

  # apply dense layer
  current = tf.keras.layers.Dense(
    units=units,
    use_bias=True,
    activation=activation,
    kernel_initializer=kernel_initializer,
    kernel_regularizer=tf.keras.regularizers.l1_l2(l1_scale, l2_scale)
    )(inputs)

  return current

class StochasticReverseComplement(tf.keras.layers.Layer):
  """Stochastically reverse complement a one hot encoded DNA sequence."""
  def __init__(self, **kwargs):
    super(StochasticReverseComplement, self).__init__(**kwargs)
  def call(self, seq_1hot, training=None):
    # if training:
    #   rc_seq_1hot = tf.gather(seq_1hot, [3, 2, 1, 0], axis=-1)
    #   rc_seq_1hot = tf.reverse(rc_seq_1hot, axis=[1])
    #   reverse_bool = tf.random.uniform(shape=[]) > 0.5
    #   src_seq_1hot = tf.cond(reverse_bool, lambda: rc_seq_1hot, lambda: seq_1hot)
    #   return src_seq_1hot, reverse_bool
    # else:
    #   return seq_1hot, tf.constant(False)
    return seq_1hot, tf.constant(False)


class StochasticShift(tf.keras.layers.Layer):
  """Stochastically shift a one hot encoded DNA sequence."""
  def __init__(self, shift_max=0, pad='uniform',  **kwargs):
    super(StochasticShift, self).__init__(**kwargs)
    self.shift_max = shift_max
    self.augment_shifts = tf.range(-self.shift_max, self.shift_max+1)
    self.pad = pad

  def call(self, seq_1hot, training=None):
    # if training:
    #   shift_i = tf.random.uniform(shape=[], minval=0, dtype=tf.int64,
    #                               maxval=len(self.augment_shifts))
    #   shift = tf.gather(self.augment_shifts, shift_i)
    #   sseq_1hot = tf.cond(tf.not_equal(shift, 0),
    #                       lambda: shift_sequence(seq_1hot, shift),
    #                       lambda: seq_1hot)
    #   return sseq_1hot
    # else:
    return seq_1hot

  def get_config(self):
    config = super().get_config().copy()
    config.update({
      'shift_max': self.shift_max,
      'pad': self.pad
    })
    return config

class SwitchReverse(tf.keras.layers.Layer):
  """Reverse predictions if the inputs were reverse complemented."""
  def __init__(self, **kwargs):
    super(SwitchReverse, self).__init__(**kwargs)
  def call(self, x_reverse):
    x = x_reverse[0]
    reverse = x_reverse[1]

    xd = len(x.shape)
    if xd == 3:
      rev_axes = [1]
    elif xd == 4:
      rev_axes = [1,2]
    else:
      raise ValueError('Cannot recognize SwitchReverse input dimensions %d.' % xd)

    return tf.keras.backend.switch(reverse,
                                   tf.reverse(x, axis=rev_axes),
                                   x)


def shift_sequence(seq, shift, pad_value=0.25):
  """Shift a sequence left or right by shift_amount.

  Args:
  seq: [batch_size, seq_length, seq_depth] sequence
  shift: signed shift value (tf.int32 or int)
  pad_value: value to fill the padding (primitive or scalar tf.Tensor)
  """
  # seq = np.squeeze(seq)
  if seq.shape.ndims != 3:
      print(seq.shape.ndims)
      raise ValueError('input sequence should be rank 3')
  input_shape = seq.shape

  pad = pad_value * tf.ones_like(seq[:, 0:tf.abs(shift), :])

  def _shift_right(_seq):
    # shift is positive
    sliced_seq = _seq[:, :-shift:, :]
    return tf.concat([pad, sliced_seq], axis=1)

  def _shift_left(_seq):
    # shift is negative
    sliced_seq = _seq[:, -shift:, :]
    return tf.concat([sliced_seq, pad], axis=1)

  sseq = tf.cond(tf.greater(shift, 0),
                 lambda: _shift_right(seq),
                 lambda: _shift_left(seq))
  sseq.set_shape(input_shape)
  # sseq = np.expand_dims(sseq, axis=-1)
  return sseq

def activate(current, activation, verbose=False):
  if verbose: print('activate:',activation)
  if activation == 'relu':
    current = tf.keras.layers.ReLU()(current)
  elif activation == 'polyrelu':
    current = PolyReLU()(current)
  elif activation == 'gelu':
    current = GELU()(current)
  elif activation == 'sigmoid':
    current = tf.keras.layers.Activation('sigmoid')(current)
  elif activation == 'tanh':
    current = tf.keras.layers.Activation('tanh')(current)
  elif activation == 'exp':
    current = Exp()(current)
  elif activation == 'softplus':
    current = Softplus()(current)
  else:
    print('Unrecognized activation "%s"' % activation, file=sys.stderr)
    exit(1)

  return current

class GELU(tf.keras.layers.Layer):
    def __init__(self, name=None, **kwargs):
        super(GELU, self).__init__(**kwargs)
    def call(self, x):
        # return tf.keras.activations.sigmoid(1.702 * x) * x
        return tf.keras.activations.sigmoid(tf.constant(1.702) * x) * x

def bpnet(input_shape, output_shape, strand_num=1, filtN_1=64, filtN_2=64,
          kern_1=25, kern_2=3, kern_3=25):
    # filtN_2 [64, 128, 256]
    # filtN_1 [64, 128]
    # trnaspose kernel_size [7, 17, 25]
    window_size = int(input_shape[0]/output_shape[0])
    #body
    input = keras.layers.Input(shape=input_shape)
    x = keras.layers.Conv1D(filtN_1, kernel_size=kern_1, padding='same',
                            activation='relu')(input)
    for i in range(1,10):
        conv_x = keras.layers.Conv1D(filtN_2,kernel_size=kern_2, padding='same',
                                     activation='relu', dilation_rate=2**i)(x)
        x = keras.layers.Add()([conv_x,x])

    bottleneck = x

    #heads
    outputs = []
    for task in range(0,output_shape[1]):
        px = keras.layers.Reshape((-1,1,filtN_2))(bottleneck)
        px = keras.layers.Conv2DTranspose(strand_num,kernel_size=(kern_3, 1),
                                          padding='same')(px)
        px = keras.layers.Reshape((-1,strand_num))(px)
        px = keras.layers.AveragePooling1D(pool_size=window_size, strides=None,
                                           padding='valid')(px)
        outputs.append(px)

    outputs = tf.keras.layers.concatenate(outputs)
    #outputs = keras.layers.Reshape((output_shape))(outputs)
    model = keras.models.Model([input],outputs)
    return model

def ori_bpnet(input_shape, output_shape, strand_num=1, filtN_1=64, filtN_2=64,
          kern_1=25, kern_2=3, kern_3=25):

    #body
    window_size = int(input_shape[0]/output_shape[0])
    input = keras.layers.Input(shape=input_shape)
    x = keras.layers.Conv1D(filtN_1, kernel_size=kern_1, padding='same',
                            activation='relu')(input)
    for i in range(1,10):
        conv_x = keras.layers.Conv1D(filtN_2,kernel_size=kern_2, padding='same',
                                     activation='relu', dilation_rate=2**i)(x)
        x = keras.layers.Add()([conv_x,x])

    bottleneck = x

    #heads
    profile_outputs = []
    count_outputs = []
    for task in range(0,output_shape[1]):
        #profile shape head
        px = keras.layers.Reshape((-1,1,filtN_2))(bottleneck)
        px = keras.layers.Conv2DTranspose(strand_num,kernel_size=(kern_3, 1),
                                          padding='same')(px)
        px = keras.layers.Reshape((-1,strand_num))(px)
        px = keras.layers.AveragePooling1D(pool_size=window_size, strides=None,
                                           padding='valid')(px)
        profile_outputs.append(px)
        #total counts head
        cx = keras.layers.GlobalAvgPool1D()(bottleneck)
        count_outputs.append(keras.layers.Dense(strand_num)(cx))

    profile_outputs = tf.keras.layers.concatenate(profile_outputs)

    count_outputs = tf.keras.layers.concatenate(count_outputs)
    model = keras.models.Model([input],[profile_outputs,count_outputs])
    return model

def lstm(input_shape,output_shape):
    input_layer= keras.layers.Input(input_shape)
    conv1 = conv_layer(input_layer,
                       num_filters=64,
                       kernel_size=25,
                       padding='same',
                       activation='exponential',
                       dropout=0.1,
                       l2=1e-5)
    conv1_residual = dilated_residual_block(conv1,
                                            num_filters=64,
                                            filter_size=7,
                                            activation='relu',
                                            l2=1e-6)
    conv1_residual_pool = keras.layers.MaxPool1D(pool_size=10,
                                                    strides=5,
                                                    padding='same'
                                                    )(conv1_residual)
    conv1_residual_dropout = keras.layers.Dropout(0.1)(conv1_residual_pool)




    conv2 = conv_layer(conv1_residual,
                       num_filters= 64,
                       kernel_size=11,
                       padding='same',
                       activation='relu',
                       dropout=0.1,
                       l2=1e-6)
    conv2_residual =dilated_residual_block2(conv2,
                                            num_filters=64,
                                            filter_size=11,
                                            activation='relu',
                                            l2=1e-6)
    conv2_residual_pool = keras.layers.MaxPool1D(pool_size=25, #TODO: make larger - 25
                                                    strides=5,
                                                    padding='same'
                                                    )(conv2_residual)
    conv2_residual_dropout = keras.layers.Dropout(0.2)(conv2_residual_pool)



    bi_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128))(conv1_residual)
    #TODO: add downsampling - global avg pool
    avg_pool = tf.expand_dims(tf.reduce_mean(bi_lstm, axis=1),-1)
    nn = tf.keras.layers.Dense(output_shape[0]*output_shape[1],activation = 'relu')(avg_pool)
    gelu_layer = GELU()
    nn = gelu_layer(nn)
    output = tf.keras.layers.Reshape(output_shape)(nn)


    model = keras.Model(inputs=input_layer, outputs=output)

    return model




def conv_layer(inputs, num_filters, kernel_size, padding='same', activation='relu', dropout=0.2, l2=None):
    if not l2:
        l2 = keras.regularizers.l2(l2)
    else:
        l2 = None

    conv1 = keras.layers.Conv1D(filters=num_filters,
                           kernel_size=kernel_size,
                           strides=1,
                           activation=None,
                           use_bias=False,
                           padding=padding,
                           kernel_initializer='glorot_normal',
                           kernel_regularizer=l2,
                           bias_regularizer=None,
                           activity_regularizer=None,
                           kernel_constraint=None,
                           bias_constraint=None,
                           )(inputs)
    conv1_bn = keras.layers.BatchNormalization()(conv1)
    conv1_active = keras.layers.Activation(activation)(conv1_bn)
    conv1_dropout = keras.layers.Dropout(dropout)(conv1_active)
    return conv1_dropout

def dilated_residual_block(input_layer, num_filters, filter_size, activation='relu', l2=None):
    if not l2:
        l2 = keras.regularizers.l2(l2)
    else:
        l2 = None

    residual_conv1 = keras.layers.Conv1D(filters=num_filters,
                                   kernel_size=filter_size,
                                   strides=1,
                                   activation=None,
                                   use_bias=False,
                                   padding='same',
                                   dilation_rate=2,
                                   kernel_initializer='glorot_normal',
                                   kernel_regularizer=l2
                                   )(input_layer)
    residual_conv1_bn = keras.layers.BatchNormalization()(residual_conv1)
    residual_conv1_active = keras.layers.Activation('relu')(residual_conv1_bn)
    residual_conv1_dropout = keras.layers.Dropout(0.1)(residual_conv1_active)
    residual_conv2 = keras.layers.Conv1D(filters=num_filters,
                                   kernel_size=filter_size,
                                   strides=1,
                                   activation=None,
                                   use_bias=False,
                                   padding='same',
                                   dilation_rate=4,
                                   kernel_initializer='glorot_normal',
                                   kernel_regularizer=l2
                                   )(residual_conv1_dropout)
    residual_conv2_bn = keras.layers.BatchNormalization()(residual_conv2)
    residual_conv2_active = keras.layers.Activation('relu')(residual_conv2_bn)
    residual_conv2_dropout = keras.layers.Dropout(0.1)(residual_conv2_active)
    residual_conv3 = keras.layers.Conv1D(filters=num_filters,
                                   kernel_size=filter_size,
                                   strides=1,
                                   activation=None,
                                   use_bias=False,
                                   padding='same',
                                   dilation_rate=8,
                                   kernel_initializer='glorot_normal',
                                   kernel_regularizer=l2
                                   )(residual_conv2_dropout)
    residual_conv3_bn = keras.layers.BatchNormalization()(residual_conv3)
    residual_sum = keras.layers.add([input_layer, residual_conv3_bn])
    return keras.layers.Activation(activation)(residual_sum)



def dilated_residual_block2(input_layer, num_filters, filter_size, activation='relu', l2=None):
    if not l2:
        l2 = keras.regularizers.l2(l2)
    else:
        l2 = None

    residual_conv1 = keras.layers.Conv1D(filters=num_filters,
                                   kernel_size=filter_size,
                                   strides=1,
                                   activation=None,
                                   use_bias=False,
                                   padding='same',
                                   dilation_rate=2,
                                   kernel_initializer='glorot_normal',
                                   kernel_regularizer=l2
                                   )(input_layer)
    residual_conv1_bn = keras.layers.BatchNormalization()(residual_conv1)
    residual_conv1_active = keras.layers.Activation('relu')(residual_conv1_bn)
    residual_conv1_dropout = keras.layers.Dropout(0.1)(residual_conv1_active)
    residual_conv2 = keras.layers.Conv1D(filters=num_filters,
                                   kernel_size=filter_size,
                                   strides=1,
                                   activation=None,
                                   use_bias=False,
                                   padding='same',
                                   dilation_rate=2,
                                   kernel_initializer='glorot_normal',
                                   kernel_regularizer=l2
                                   )(residual_conv1_dropout)
    residual_conv2_bn = keras.layers.BatchNormalization()(residual_conv2)
    residual_sum = keras.layers.add([input_layer, residual_conv2_bn])
    return keras.layers.Activation(activation)(residual_sum)
