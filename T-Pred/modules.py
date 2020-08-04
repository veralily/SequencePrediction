from __future__ import print_function
import tensorflow as tf
import numpy as np

def embedding(inputs,
              vocab_size,
              num_units,
              scale=True,
              scope='embedding',
              with_t='True',
              reuse=None):
    """Embeds a given tensor
    Args:
        inputs: A Tensor with type `int32` or `int64` containing the IDs
        vocab_size: An int.
        num_units: An int. Number of embedding hidden units.
        scale: A boolean. If True, the outputs is multiples by sqrt num_units.
        scope: Optional scope for `variable_scope`.
        reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
        A `Tensor` with one more rank than inputs's. The last dimensionality
        should be `num_units`.
        """
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable('lookup_table',
                                       dtype=tf.float32,
                                       shape=[vocab_size,num_units])
        outputs = tf.nn.embedding_lookup(lookup_table, inputs)

        if scale:
            outputs = outputs * (num_units ** 0.5)
        if with_t:
            return outputs, lookup_table
        else:
            return outputs


def multihead_attention(queries,
                        keys,
                        num_units,
                        num_heads=4,
                        drop_rate=0,
                        is_training=True,
                        causality=False,
                        residual=True,
                        scope='multihead_attention',
                        reuse=None,
                        with_qk=False):
    """

    :param queries: A 3d tensor with shape of [N, T_q, C_q].
    :param keys: A 3d tensor with shape of [N, T_k, C_k].
    :param num_units: A scalar. Attention size.
    :param num_heads: An int. Number of heads.
    :param drop_rate: A floating point number.
    :param is_training: Boolean. Controller of mechanism for dropout.
    :param causality: Boolean. If true, units that reference the future are masked.
    :param scope: Optional scope for `variable_scope`.
    :param reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    :param with_qk:
    :return: A 3d tensor with shape of (N, T_q, C).
    """
    with tf.variable_scope(scope, reuse=reuse):
        if num_units is None:
            num_units = queries.get_shape().as_list()[-1]
        Q = tf.layers.dense(queries, num_units, activation=None)
        K = tf.layers.dense(keys, num_units, activation=None)
        V = tf.layers.dense(keys, num_units, activation=None)

        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
        V_ = tf.concat(tf.split(V, num_units, axis=2), axis=0)

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))
        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # Optional Mask: future blinding to contain causality
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :])
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])

            paddings = tf.ones_like(masks)*(-2**32+1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)

        # Activation: softmax
        outputs = tf.nn.softmax(outputs)

        # Dropout
        outputs = tf.layers.dropout(outputs, rate=drop_rate, training=tf.convert_to_tensor(is_training))

        # Weighted sum: output
        outputs = tf.matmul(outputs, V_) # [N*h, T_q, C/h]

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)

        # Residual connection
        if residual:
            outputs += queries

        if with_qk:
            return Q, K
        else:
            return outputs


def conv1d(inputs,
           num_units,
           scope='Conv1D',
           reuse=True,
           kernel_size=3,
           strides=1,
           padding='same',
           data_format='channels_last',
           activation='relu'):
    """

    :param inputs: A Tensor. (N, T, C)
    :param num_units: An int. The number of filters.
    :param scope: A str.
    :param reuse: Boolean.
    :param kernel_size: The width of filter.
    :param strides: The stride for the window.
    :param padding: Padding mode.
    :param data_format: "Chanel last"
    :param activation: Activation mode
    :return: A Tensor. (N, T(maybe less than T, up to padding mode), num_units)
    """
    with tf.variable_scope(scope, reuse=reuse):
        outputs = tf.keras.layers.Conv1D(filters=num_units,
                                         kernel_size=kernel_size,
                                         strides=strides,
                                         padding=padding,
                                         data_format=data_format,
                                         activation=activation)(inputs)
    return outputs


def res_block(inputs,
              num_units,
              res_rate,
              scope = 'ResBlock',
              reuse=True):
    """

    :param inputs: A Tensor. (N, T, C)
    :param num_units: An int.
    :param res_rate: An int.
    :param scope: A Str.
    :param reuse: Boolean.
    :return: A Tensor. (N, T, num_units)
    """
    with tf.variable_scope(scope, reuse=reuse):
        output = inputs
        output = tf.nn.relu(output)
        output = conv1d(output, scope= 'Conv1D.1', reuse=True, num_units=num_units)
        output = tf.nn.relu(output)
        output = conv1d(output, scope= 'Conv1D.2', reuse=True, num_units=num_units)
    return inputs + (res_rate * output)

def gumbel_softmax(inputs, tau=1e-4, axis=-1):
    """Samples a tensor from a Gumbel distribution.
    Args:
        inputs: A tensor to be calculated. (N, T, n)
        tau: Temperature parameter which controls the smoothness of x.
    Returns:
        x: A Tensor. gumbel softmax (N, T, n)
        y: An int. The argmax of x. (N, T)
    """
    EPSILON = 1e-20

    # Samples an uniform distribution based on the input shape
    uniform_dist = tf.random.uniform(inputs.get_shape(), 0, 1)

    # Samples from the Gumbel distribution
    gumbel_dist = -1 * tf.math.log(-1 * tf.math.log(uniform_dist + EPSILON) + EPSILON)

    # Adds a sampled Gumbel distribution to the input
    x = inputs + gumbel_dist

    # Applying the softmax over the Gumbel-based input
    x = tf.nn.softmax(x / tau, axis=axis)

    # Sampling an argmax token from the Gumbel-based input
    y = tf.stop_gradient(tf.argmax(x, axis=axis))

    return x, y