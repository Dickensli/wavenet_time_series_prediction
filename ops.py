from __future__ import division

import tensorflow as tf


def create_adam_optimizer(learning_rate, momentum):
    return tf.train.AdamOptimizer(learning_rate=learning_rate,
                                  epsilon=1e-4)


def create_sgd_optimizer(learning_rate, momentum):
    return tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                      momentum=momentum)


def create_rmsprop_optimizer(learning_rate, momentum):
    return tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                     momentum=momentum,
                                     epsilon=1e-5)


optimizer_factory = {'adam': create_adam_optimizer,
                     'sgd': create_sgd_optimizer,
                     'rmsprop': create_rmsprop_optimizer}


def time_to_batch(value, dilation, name=None):
    '''
    value : [batch, time, features]
    return : [batch * dilation, num_strides, features]
    '''
    with tf.name_scope('time_to_batch'):
        shape = tf.shape(value)
        pad_elements = dilation - 1 - (shape[1] + dilation - 1) % dilation
        padded = tf.pad(value, [[0, 0], [0, pad_elements], [0, 0]])
        # [batch, num_strides * dilation, features] -> [batch * num_strides, dilation, features]
        reshaped = tf.reshape(padded, [-1, dilation, shape[2]])
        # [batch * num_strides, dilation, features] -> [dilation, batch * num_strides, features]
        transposed = tf.transpose(reshaped, perm=[1, 0, 2])
        # [dilation, batch * num_strides, features] -> [batch * dilation, num_strides, features]
        # apply dense conv (size=2) on dim 1
        return tf.reshape(transposed, [shape[0] * dilation, -1, shape[2]])


def batch_to_time(value, dilation, name=None):
    '''
    value : [batch * dilation, num_strides, features]
    return : [batch, time, features]
    '''
    with tf.name_scope('batch_to_time'):
        shape = tf.shape(value)
        # [batch * dilation, num_strides, features] -> [dilation, batch * num_strides, features]
        prepared = tf.reshape(value, [dilation, -1, shape[2]])
        # [dilation, batch * num_strides, features] -> [batch * num_strides, dilation, features]
        transposed = tf.transpose(prepared, perm=[1, 0, 2])
        # [batch * num_strides, dilation, features] -> [batch, num_strides * dilation, features]
        # apply dense conv (size=2) on dim 0
        return tf.reshape(transposed,
                          [tf.div(shape[0], dilation), -1, shape[2]])


def causal_conv(value, filter_, dilation, name='causal_conv'):
    '''
    value : [batch, time, features]
    
    '''
    with tf.name_scope(name):
        filter_width = tf.shape(filter_)[0]
        if dilation > 1:
            # [batch, time, features] -> [batch * dilation, num_strides, features]
            transformed = time_to_batch(value, dilation)
            # NWC -> NWC
            conv = tf.nn.conv1d(transformed, filter_, stride=1,
                                padding='VALID')
            # [batch * dilation, num_strides, features] -> [batch, time, features]
            restored = batch_to_time(conv, dilation)
        else:
            restored = tf.nn.conv1d(value, filter_, stride=1, padding='VALID')
        # Remove excess elements at the end.
        out_width = tf.shape(value)[1] - (filter_width - 1) * dilation
        # [batch, time, features]
        result = tf.slice(restored,
                          [0, 0, 0],
                          [-1, out_width, -1])
        return result



def mu_law_encode(audio, quantization_channels):
    '''Quantizes waveform amplitudes.'''
    with tf.name_scope('encode'):
        mu = tf.to_float(quantization_channels - 1)
        # Perform mu-law companding transformation (ITU-T, 1988).
        # Minimum operation is here to deal with rare large amplitudes caused
        # by resampling.
        safe_audio_abs = tf.minimum(tf.abs(audio), 1.0)
        magnitude = tf.log1p(mu * safe_audio_abs) / tf.log1p(mu)
        signal = tf.sign(audio) * magnitude
        # Quantize signal to the specified number of levels.
        return tf.to_int32((signal + 1) / 2 * mu + 0.5)



def sequence_dense_layer(inputs,output_units, bias=True, activation=None, batch_norm=None,
                                 dropout=None, scope='sequence_dense_layer', reuse=False):
    """
    Applies a shared dense layer to each timestep of a tensor of shape [batch_size, max_seq_len, input_units]
    to produce a tensor of shape [batch_size, max_seq_len, output_units]. FC which in wavenet is a conveltion layer

    Args:
        inputs: Tensor of shape [batch size, max sequence length, ...].
        output_units: Number of output units.
        activation: activation function.
        dropout: dropout keep prob.

    Returns:
        Tensor of shape [batch size, max sequence length, output_units].
    """
    #reuse=false;means this scope should use once
    with tf.variable_scope(scope, reuse=reuse):
        W = tf.get_variable(
            name='weights',
            initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0 / float(shape(inputs,-1))),
            shape=[shape(inputs,-1), output_units]
        )
        z = tf.einsum('ijk,kl->ijl', inputs, W)
        if bias:
            b = tf.get_variable(
                name='biases',
                initializer=tf.constant_initializer(),
                shape=[output_units]
            )
            z = z + b

        if batch_norm is not None:
            z = tf.layers.batch_normalization(z, training=batch_norm, reuse=reuse)

        z = activation(z) if activation else z
        z = tf.nn.dropout(z, dropout) if dropout is not None else z
        return z

def shape(tensor, dim=None):
    """Get tensor shape/dimension as list/int"""
    shape = tensor.get_shape()
    if dim is None:
        return shape.as_list()
    else:
        return shape.as_list()[dim]


def create_variable(name, shape):
    '''Create a convolution filter variable with the specified name and shape,
    and initialize it using Xavier initialition.'''
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    #variable = tf.Variable(, name=name)
    variable =tf.get_variable(name,shape,initializer=initializer)
    return variable

def create_bias(name,shape):
    initializer = tf.constant_initializer(value=0.0, dtype=tf.float32)
    #variable = tf.Variable(initializer(shape=shape), name=name)
    variable = tf.get_variable(name,shape,initializer=initializer)
    return variable