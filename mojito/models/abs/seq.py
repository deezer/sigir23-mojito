import tensorflow as tf

from mojito.models.model import Model
from mojito.models.core import embedding


class AbsSeqRec(Model):
    """
    Abstract sequential recommendation model
    """
    def __init__(self, sess, params, n_users, n_items):
        super(AbsSeqRec, self).__init__(sess, params,
                                        n_users, n_items)
        # boolean to check if need to scale the input
        self.input_scale = params['model']['params'].get(
            'input_scale', False)
        self.seq = None

    def build_feedict(self, batch, is_training=True):
        raise NotImplementedError('build_feedict method should be '
                                  'implemented in concrete model')

    def export_embeddings(self):
        raise NotImplementedError('export_embeddings method should be '
                                  'implemented in concrete model')

    def _create_placeholders(self):
        super(AbsSeqRec, self)._create_placeholders()
        # batch of history sequence item ids (sequence of items
        # have interactions with corresponding users
        # presented in user_ids before that user interacts with
        # the corresponding positive item)
        self.seq_ids = tf.compat.v1.placeholder(name='seq_ids',
                                                dtype=tf.int32,
                                                shape=[None, self.seqlen])
        # batch of positive item ids (items have interactions with
        # corresponding users presented in user_ids).
        # pos_ids is a shift version of seq_ids
        # for example: seq_ids = (s1, s2, s3), pos_ids = (s2, s3, s4)
        self.pos_ids = tf.compat.v1.placeholder(name='pos_ids',
                                                dtype=tf.int32,
                                                shape=[None, self.seqlen])
        # batch of negative item ids (items do not have
        # interactions with users presented in user_ids)
        self.neg_ids = tf.compat.v1.placeholder(name='neg_ids',
                                                dtype=tf.int32,
                                                shape=[None, self.seqlen])

    def _create_variables(self, reuse=None):
        """
        Build variables
        :return:
        """
        self.logger.debug('--> Create embedding tables')
        with tf.compat.v1.variable_scope('embedding_tables',
                                         reuse=reuse):
            # item embeddings
            # first row is zero embedding for padding item
            self.item_embedding_table, self.org_item_embedding_table = \
                embedding(vocab_size=self.n_items,
                          embedding_dim=self.embedding_dim,
                          zero_pad=True,
                          use_reg=self.use_reg,
                          l2_reg=self.l2_emb,
                          scope='item_embedding_table',
                          reuse=reuse)

    def _create_inference(self, name, reuse=None):
        """
        Build inference graph
        :return:
        """
        self._create_posneg_emb_inference()
        self._create_net_inference(name, reuse)
        # reshape input sequence to have the same shape as
        # pos_emb and neg_emb
        self.seq_emb = tf.reshape(
            self.seq,
            [tf.shape(self.seq_ids)[0] * self.seqlen,
             self.embedding_dim])

    def _create_posneg_emb_inference(self):
        self.logger.debug('--> Create POS/NEG inference')
        # positive output (target)
        pos_ids = tf.reshape(self.pos_ids,
                             [tf.shape(self.seq_ids)[0] * self.seqlen])
        # negative output (non-target)
        neg_ids = tf.reshape(self.neg_ids,
                             [tf.shape(self.seq_ids)[0] * self.seqlen])
        # lookup embedding for positive items
        self.pos_emb = tf.nn.embedding_lookup(
            self.item_embedding_table, pos_ids)
        # lookup embedding for negative items
        self.neg_emb = tf.nn.embedding_lookup(
            self.item_embedding_table, neg_ids)
        # ignore padding items (0)
        self.istarget = tf.reshape(tf.compat.v1.to_float(tf.not_equal(pos_ids, 0)),
                                   [tf.shape(self.seq_ids)[0] * self.seqlen])

    def _create_test_inference(self, name, reuse=None):
        # lookup embedding for test items
        # shape = [bs, ntest, dim]
        test_item_emb = tf.nn.embedding_lookup(self.item_embedding_table,
                                               self.test_item_ids)
        # dot product is used to measure affinity
        self.test_logits = tf.matmul(self.seq,
                                     tf.transpose(test_item_emb, perm=[0, 2, 1]))
        # test logits shape: [batch, seq_len, n_test_items]
        # take only the last item for the most recent interaction
        self.test_logits = self.test_logits[:, -1, :]

    def _create_net_inference(self, name, reuse=None):
        with tf.compat.v1.variable_scope(f'{name}_net_inference',
                                         reuse=reuse):
            # input sequence
            self.seq = tf.nn.embedding_lookup(self.item_embedding_table,
                                              self.seq_ids)
            # scale
            if self.input_scale is True:
                self.logger.info('Scale input sequence')
                self.seq = self.seq * (self.embedding_dim ** 0.5)
            else:
                self.logger.info('DO NOT scale input')

    def _create_loss(self):
        """
        Build loss graph
        :return:
        """
        raise NotImplementedError('_create_loss method should be '
                                  'implemented in concrete model')
