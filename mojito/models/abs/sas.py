import tensorflow as tf

from mojito.models.abs.seq import AbsSeqRec


class AbsSasRec(AbsSeqRec):
    """
    Abstract self-attention sequential recommendation model
    """
    def __init__(self, sess, params, n_users, n_items):
        super(AbsSasRec, self).__init__(sess, params,
                                        n_users, n_items)
        self.num_blocks = params['model']['params'].get(
            'num_blocks', 2)
        self.num_heads = params['model']['params'].get(
            'num_heads', 1)
        self.causality = params['model']['params'].get(
            'causality', True)
        self.kqactivation = self._activation(params['model']['params'].get(
            'kqactivation', 'None'))

    def build_feedict(self, batch, is_training=True):
        raise NotImplementedError('build_feedict method should be '
                                  'implemented in concrete model')

    def export_embeddings(self):
        item_embeddings = self.sess.run(self.item_embedding_table)
        return item_embeddings

    def _get_mask(self):
        return tf.expand_dims(tf.compat.v1.to_float(tf.not_equal(
            self.seq_ids, 0)), -1)

    def _learnable_abs_position_embedding(self, position_embedding_table):
        """
        Lookup embedding for positions
        :param position_embedding_table:
        :return:
        """
        position_ids = tf.tile(
            tf.expand_dims(tf.range(tf.shape(self.seq_ids)[1]), 0),
            [tf.shape(self.seq_ids)[0], 1])
        position = tf.nn.embedding_lookup(position_embedding_table,
                                          position_ids)
        return position

    def _create_loss(self):
        """
        Build loss graph
        :return:
        """
        # prediction layer
        self.pos_logits = tf.reduce_sum(self.pos_emb * self.seq_emb, -1)
        self.neg_logits = tf.reduce_sum(self.neg_emb * self.seq_emb, -1)
        self.loss = tf.reduce_sum(
            - tf.math.log(tf.sigmoid(self.pos_logits) + 1e-24) * self.istarget -
            tf.math.log(1 - tf.sigmoid(self.neg_logits) + 1e-24) * self.istarget
        ) / tf.reduce_sum(self.istarget)
        self.reg_loss = sum(tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES))
        self.loss += self.reg_loss
