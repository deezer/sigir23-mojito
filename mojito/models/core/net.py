import tensorflow as tf


def embedding(vocab_size,
              embedding_dim,
              zero_pad=True,
              l2_reg=0.0,
              scope='embedding',
              use_reg=True,
              initializer=None,
              reuse=None):
    """
    Create an embedding table
    :param vocab_size:
    :param embedding_dim:
    :param zero_pad:
    :param l2_reg:
    :param scope:
    :param use_reg:
    :param initializer:
    :param reuse:
    :return:
    """
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        if use_reg is True:
            if initializer is not None and initializer == 'random_normal':
                lookup_table = tf.compat.v1.get_variable(
                    f'{scope}_lookup_table',
                    dtype=tf.float32,
                    shape=[vocab_size, embedding_dim],
                    initializer=tf.random_normal_initializer(
                        0., stddev=1. / (embedding_dim ** 0.5)),
                    regularizer=tf.keras.regularizers.L2(l2_reg))
            else:
                lookup_table = tf.compat.v1.get_variable(
                    f'{scope}_lookup_table',
                    dtype=tf.float32,
                    shape=[vocab_size, embedding_dim],
                    regularizer=tf.keras.regularizers.L2(l2_reg))
        else:
            lookup_table = tf.compat.v1.get_variable(
                f'{scope}_lookup_table',
                dtype=tf.float32,
                shape=[vocab_size, embedding_dim],
                initializer=tf.random_normal_initializer(
                    0., stddev=1. / (embedding_dim ** 0.5)))
        # zero pad at the beginning of the table (index = 0)
        if zero_pad:
            padded_lookup_table = tf.concat((tf.zeros(shape=[1, embedding_dim]),
                                             lookup_table), axis=0)
            return padded_lookup_table, lookup_table
    return lookup_table


def normalize(inputs,
              epsilon=1e-6,
              scope='ln',
              reuse=None):
    """
    Layer normalization
    :param inputs: A tensor with 2 or more dimensions, where the
                   first dimension has `batch_size`
    :param epsilon: A floating number. A very small number for
                    preventing Zero Division Error
    :param scope: Optional scope for `variable_scope`
    :param reuse: Boolean, whether to reuse the weights of a
                  previous layer by the same name
    :return: A tensor with the same shape and data dtype as `inputs`
    """
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keepdims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** 0.5)
        outputs = gamma * normalized + beta
    return outputs


def feedforward(inputs,
                num_units=(2048, 512),
                scope='multihead_attention',
                dropout_rate=0.2,
                is_training=True,
                reuse=None,
                normalized=False):
    """
    Feed forwork network component
    :param inputs: A 3d tensor with shape of [N, T, C]
    :param num_units: A list of two integers
    :param scope: Optional scope for `variable_scope`
    :param dropout_rate:
    :param is_training: Boolean. Dropout controller
    :param reuse: Boolean, whether to reuse the weights of a
                  previous layer by the same name
    :param normalized: Boolean, whether to normalize the output
    :return: A 3d tensor with the same shape and dtype as inputs
    """
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0],
                  "kernel_size": 1, "activation": tf.nn.relu,
                  "use_bias": True}
        outputs = tf.compat.v1.layers.conv1d(**params)
        outputs = tf.compat.v1.layers.dropout(
            outputs,
            rate=dropout_rate,
            training=tf.convert_to_tensor(is_training))
        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1],
                  "kernel_size": 1, "activation": None,
                  "use_bias": True}
        outputs = tf.compat.v1.layers.conv1d(**params)
        outputs = tf.compat.v1.layers.dropout(
            outputs,
            rate=dropout_rate,
            training=tf.convert_to_tensor(is_training))
        # Residual connection
        if outputs.shape.as_list()[-1] == inputs.shape.as_list()[-1]:
            outputs += inputs
        # Normalize
        if normalized:
            outputs = normalize(outputs)
    return outputs


def add_weight(dimension, name='weight'):
    """
    Create a weight variable
    :param dimension:
    :param name:
    :return:
    """
    with tf.compat.v1.variable_scope(name,
                                     reuse=tf.compat.v1.AUTO_REUSE):
        w = tf.compat.v1.get_variable(
            name=name,
            dtype=tf.float32,
            shape=[dimension, dimension],
            initializer=tf.random_normal_initializer(
                0., stddev=1. / (dimension ** 0.5)))
    return w
