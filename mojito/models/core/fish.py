import tensorflow as tf

from mojito import MojitoError
from mojito.models.core.net import feedforward, normalize

"""
This code is based on the original pytorch code found in
https://github.com/minhtannguyen/FishFormer
Reference: Nguyen, Tan, et al. "Improving transformer with an admixture of attention heads." 
Advances in Neural Information Processing Systems 35 (2022): 27937-27952.
"""


def hdp_net(inputs, input_dim, output_dim,
            activation='relu'):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(input_dim,)))
    model.add(tf.keras.layers.Dense(output_dim,
                                    activation=activation))
    model.add(tf.keras.layers.Dense(output_dim))
    return model(inputs)


def multihead_attention(queries_list,
                        keys_list,
                        sigma_noise,
                        num_heads=8,
                        dim_head=16,
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        scope="multihead_attention",
                        residual_type='add',
                        reuse=None,
                        with_att=False):
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        num_global_heads = len(queries_list)
        queries_it, queries_ctx = queries_list
        keys_it, keys_ctx = keys_list
        dim_glob_head = dim_head
        Q_glob_it = tf.compat.v1.layers.dense(queries_it, dim_glob_head,
                                              activation=None)  # (N, T_q, C)
        K_glob_it = tf.compat.v1.layers.dense(keys_it, dim_glob_head,
                                              activation=None)  # (N, T_k, C)
        Q_glob_ctx = tf.compat.v1.layers.dense(queries_ctx, dim_glob_head,
                                               activation=None)  # (N, T_q, C)
        K_glob_ctx = tf.compat.v1.layers.dense(keys_ctx, dim_glob_head,
                                               activation=None)  # (N, T_k, C)
        V = tf.compat.v1.layers.dense(keys_it, num_heads*dim_head,
                                      activation=None)  # (N, T_k, C)
        # Split and concat
        Q_glob_it_, K_glob_it_ = Q_glob_it, K_glob_it
        Q_glob_ctx_, K_glob_ctx_ = Q_glob_ctx, K_glob_ctx
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        # Attention scores
        mean_att_scores_it = tf.matmul(Q_glob_it_,
                                       tf.transpose(K_glob_it_, [0, 2, 1]))  # (h*N, T_q, T_k)
        mean_att_scores_ctx = tf.matmul(Q_glob_ctx_,
                                        tf.transpose(K_glob_ctx_, [0, 2, 1]))  # (h*N, T_q, T_k)
        mean_att_scores_list = [mean_att_scores_it, mean_att_scores_ctx]
        mean_att_scores = tf.concat(mean_att_scores_list, axis=0)
        sigma_noise = tf.expand_dims(tf.expand_dims(tf.reshape(sigma_noise, [-1]),
                                                    axis=-1), axis=-1)
        att_scores = mean_att_scores + (sigma_noise ** 2) * \
                     tf.random.normal(tf.shape(mean_att_scores))
        att_scores = tf.reshape(att_scores, shape=[-1, num_global_heads,
                                                   tf.shape(Q_glob_it)[1],
                                                   tf.shape(K_glob_it)[1]])
        att_scores = tf.transpose(att_scores, perm=[0, 2, 3, 1])
        att_scores = tf.reshape(att_scores, shape=[-1, num_global_heads])
        att_scores = hdp_net(att_scores, input_dim=num_global_heads,
                             output_dim=num_heads)
        att_scores = tf.transpose(tf.reshape(
            att_scores,
            shape=[-1, tf.shape(Q_glob_it)[1], tf.shape(K_glob_it)[1], num_heads]),
            perm=[0, 3, 1, 2])
        att_scores = tf.reshape(att_scores, shape=[-1, tf.shape(Q_glob_it)[1],
                                                   tf.shape(K_glob_it)[1]])
        # Scale
        att_scores = att_scores / (dim_head ** 0.5)

        # Key Masking
        key_masks = tf.sign(tf.reduce_sum(tf.abs(keys_it), axis=-1))  # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1),
                            [1, tf.shape(queries_it)[1], 1])  # (h*N, T_q, T_k)

        paddings = tf.ones_like(att_scores) * (-2 ** 32 + 1)
        att_scores = tf.where(tf.equal(key_masks, 0), paddings, att_scores)  # (h*N, T_q, T_k)

        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(att_scores[0, :, :])  # (T_q, T_k)
            tril = tf.compat.v1.linalg.LinearOperatorLowerTriangular(
                diag_vals).to_dense()  # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0),
                            [tf.shape(att_scores)[0], 1, 1])  # (h*N, T_q, T_k)
            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            att_scores = tf.where(tf.equal(masks, 0),
                                  paddings, att_scores)  # (h*N, T_q, T_k)

        # Activation
        att_scores = tf.nn.softmax(att_scores)

        # Query Masking
        query_masks = tf.sign(tf.reduce_sum(tf.abs(queries_it), axis=-1))  # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1),
                              [1, 1, tf.shape(keys_it)[1]])  # (h*N, T_q, T_k)
        att_scores *= query_masks  # broadcasting. (N, T_q, C)

        # Dropouts
        att_scores = tf.compat.v1.layers.dropout(
            att_scores, rate=dropout_rate,
            training=tf.convert_to_tensor(is_training))
        # Weighted sum
        outputs = tf.matmul(att_scores, V_)  # ( h*N, T_q, C/h)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0),
                            axis=2)  # (N, T_q, C)
        # Residual connection
        if residual_type == 'add':
            outputs += queries_it
        elif residual_type == 'mult':
            outputs *= queries_it
        else:
            raise MojitoError(f'Not support residual type {residual_type}')

    if with_att:
        return outputs, att_scores
    else:
        return outputs


def multi_head_attention_blocks(seq, context_seq, num_blocks, dim_head,
                                num_heads, sigma_noise,
                                dropout_rate, mask,
                                output_dim=-1,
                                causality=True,
                                residual_type='add',
                                reuse=None,
                                is_training=False,
                                name=''):
    embedding_dim = num_heads * dim_head
    for i in range(num_blocks):
        with tf.compat.v1.variable_scope(f'{name}_num_blocks_{i}'):
            queries_list = [normalize(seq), normalize(context_seq)]
            keys_list = [seq, context_seq]
            # Self-attention
            seq = multihead_attention(
                queries_list=queries_list,
                keys_list=keys_list,
                num_heads=num_heads,
                dim_head=dim_head,
                sigma_noise=sigma_noise,
                dropout_rate=dropout_rate,
                is_training=is_training,
                causality=causality,
                residual_type=residual_type,
                reuse=reuse,
                scope=f'{name}_self_attention{i}')
            if i == num_blocks - 1 and output_dim > 0:
                num_units = [embedding_dim, output_dim]
            else:
                num_units = [embedding_dim, embedding_dim]
            # Feed forward net
            seq = feedforward(
                normalize(seq),
                num_units=num_units,
                dropout_rate=dropout_rate,
                is_training=is_training)
            seq *= mask
            context_seq = seq[:, :, dim_head:]
    return seq
