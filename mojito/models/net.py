import numpy as np
import tensorflow as tf

from mojito.models.core import normalize, embedding
from mojito.models.core.fish import \
    multi_head_attention_blocks as admix_multi_head_attention_blocks
from mojito.models.abs.sas import AbsSasRec


class Mojito(AbsSasRec):
    """
    Mixture of item & context attention based on the work:
    Nguyen et al. "Improving transformer with an admixture of attention heads."
    Neurips 2022.
    """
    def __init__(self, sess, params, n_users, n_items):
        super(Mojito, self).__init__(sess, params,
                                     n_users, n_items)
        self.tempo_embedding_dim = params.get('tempo_embedding_dim', 8)
        fism_params = params['model']['params']['fism']
        self.n_fism_elems = fism_params.get('n_items', 50)
        self.beta = fism_params.get('beta', 1.0)
        self.expand_dim = 3
        self.use_year = params['model']['params'].get('use_year', True)
        self.num_contexts = 7
        self.tempo_linspace = params.get('tempo_linspace', 8)
        self.lambda_trans_seq = params['model']['params'].get('lambda_trans_seq', 0.1)
        self.lambda_glob = params['model']['params'].get('lambda_glob', 0.1)
        self.lambda_ctx = params['model']['params'].get('lambda_ctx', 0.1)
        self.residual_type = params['model']['params'].get(
            'residual', 'add')
        # one for item, one for context
        self.num_global_heads = 2
        self.num_trans_global_heads = 2
        self.dim_head = self.embedding_dim
        self.ctx_activation = self._activation(
            params['model']['params'].get('ctx_activation', 'none'))
        self.local_output_dim = 2 * self.embedding_dim
        self.lambda_user = params['model']['params'].get('lambda_user', 0.0)
        self.lambda_item = params['model']['params'].get('lambda_item', 0.0)

    def build_feedict(self, batch, is_training=True):
        feed_dict = {
            self.user_ids: batch[0],
            self.seq_ids: batch[1],
            self.is_training: is_training
        }
        if is_training is True:
            feed_dict[self.pos_ids] = batch[2]
            feed_dict[self.neg_ids] = batch[3]
            feed_dict[self.seq_year_ids] = batch[4]
            feed_dict[self.seq_month_ids] = batch[5]
            feed_dict[self.seq_day_ids] = batch[6]
            feed_dict[self.seq_dayofweek_ids] = batch[7]
            feed_dict[self.seq_dayofyear_ids] = batch[8]
            feed_dict[self.seq_week_ids] = batch[9]
            feed_dict[self.seq_hour_ids] = batch[10]
            feed_dict[self.pos_year_ids] = batch[11]
            feed_dict[self.pos_month_ids] = batch[12]
            feed_dict[self.pos_day_ids] = batch[13]
            feed_dict[self.pos_dayofweek_ids] = batch[14]
            feed_dict[self.pos_dayofyear_ids] = batch[15]
            feed_dict[self.pos_week_ids] = batch[16]
            feed_dict[self.pos_hour_ids] = batch[17]
            feed_dict[self.item_fism_ids] = batch[18]
        else:
            feed_dict[self.test_item_ids] = batch[2]  # [N, M]
            feed_dict[self.seq_year_ids] = batch[3]
            feed_dict[self.seq_month_ids] = batch[4]
            feed_dict[self.seq_day_ids] = batch[5]
            feed_dict[self.seq_dayofweek_ids] = batch[6]
            feed_dict[self.seq_dayofyear_ids] = batch[7]
            feed_dict[self.seq_week_ids] = batch[8]
            feed_dict[self.seq_hour_ids] = batch[9]
            feed_dict[self.test_year_ids] = batch[10]
            feed_dict[self.test_month_ids] = batch[11]
            feed_dict[self.test_day_ids] = batch[12]
            feed_dict[self.test_dayofweek_ids] = batch[13]
            feed_dict[self.test_dayofyear_ids] = batch[14]
            feed_dict[self.test_week_ids] = batch[15]
            feed_dict[self.test_hour_ids] = batch[16]
            feed_dict[self.item_fism_ids] = batch[17]
        return feed_dict

    def export_embeddings(self):
        pass

    ###############################################################################
    #               INPUT PLACEHOLDERS
    ###############################################################################
    def _create_placeholders(self):
        super(Mojito, self)._create_placeholders()
        self.seq_year_ids = tf.compat.v1.placeholder(name='seq_year_ids',
                                                     dtype=tf.float32,
                                                     shape=(None, self.seqlen))
        self.seq_month_ids = tf.compat.v1.placeholder(name='seq_month_ids',
                                                      dtype=tf.float32,
                                                      shape=(None, self.seqlen))
        self.seq_day_ids = tf.compat.v1.placeholder(name='seq_day_ids',
                                                    dtype=tf.float32,
                                                    shape=(None, self.seqlen))
        self.seq_dayofweek_ids = tf.compat.v1.placeholder(name='seq_dayofweek_ids',
                                                          dtype=tf.float32,
                                                          shape=(None, self.seqlen))
        self.seq_dayofyear_ids = tf.compat.v1.placeholder(name='seq_dayofyear_ids',
                                                          dtype=tf.float32,
                                                          shape=(None, self.seqlen))
        self.seq_week_ids = tf.compat.v1.placeholder(name='seq_week_ids',
                                                     dtype=tf.float32,
                                                     shape=(None, self.seqlen))
        self.seq_hour_ids = tf.compat.v1.placeholder(name='seq_hour_ids',
                                                     dtype=tf.float32,
                                                     shape=(None, self.seqlen))
        self.pos_year_ids = tf.compat.v1.placeholder(name='pos_year_ids',
                                                     dtype=tf.float32,
                                                     shape=(None, self.seqlen))
        self.pos_month_ids = tf.compat.v1.placeholder(name='pos_month_ids',
                                                      dtype=tf.float32,
                                                      shape=(None, self.seqlen))
        self.pos_day_ids = tf.compat.v1.placeholder(name='pos_day_ids',
                                                    dtype=tf.float32,
                                                    shape=(None, self.seqlen))
        self.pos_dayofweek_ids = tf.compat.v1.placeholder(name='pos_dayofweek_ids',
                                                          dtype=tf.float32,
                                                          shape=(None, self.seqlen))
        self.pos_dayofyear_ids = tf.compat.v1.placeholder(name='pos_dayofyear_ids',
                                                          dtype=tf.float32,
                                                          shape=(None, self.seqlen))
        self.pos_week_ids = tf.compat.v1.placeholder(name='pos_week_ids',
                                                     dtype=tf.float32,
                                                     shape=(None, self.seqlen))
        self.pos_hour_ids = tf.compat.v1.placeholder(name='pos_hour_ids',
                                                     dtype=tf.float32,
                                                     shape=(None, self.seqlen))
        self.test_year_ids = tf.compat.v1.placeholder(name='test_year_ids',
                                                      dtype=tf.float32,
                                                      shape=[None])
        self.test_month_ids = tf.compat.v1.placeholder(name='test_month_ids',
                                                       dtype=tf.float32,
                                                       shape=[None])
        self.test_day_ids = tf.compat.v1.placeholder(name='test_day_ids',
                                                     dtype=tf.float32,
                                                     shape=[None])
        self.test_dayofweek_ids = tf.compat.v1.placeholder(name='test_dayofweek_ids',
                                                           dtype=tf.float32,
                                                           shape=[None])
        self.test_dayofyear_ids = tf.compat.v1.placeholder(name='test_dayofyear_ids',
                                                           dtype=tf.float32,
                                                           shape=[None])
        self.test_week_ids = tf.compat.v1.placeholder(name='test_week_ids',
                                                      dtype=tf.float32,
                                                      shape=[None])
        self.test_hour_ids = tf.compat.v1.placeholder(name='test_hour_ids',
                                                      dtype=tf.float32,
                                                      shape=[None])
        self.item_fism_ids = tf.compat.v1.placeholder(
            name='item_fism_elem_ids', dtype=tf.int32,
            shape=[None, self.n_fism_elems])

    ###############################################################################
    #               CREATE VARIABLES
    ###############################################################################
    def _create_variables(self, reuse=None):
        super(Mojito, self)._create_variables(reuse=reuse)
        with tf.compat.v1.variable_scope('embedding_tables',
                                         reuse=reuse):
            # user embedding table
            self.user_embedding_table, _ = embedding(vocab_size=self.n_users + 1,
                                                     embedding_dim=self.embedding_dim,
                                                     zero_pad=True,
                                                     use_reg=self.use_reg,
                                                     l2_reg=self.l2_emb,
                                                     scope='user_embedding_table',
                                                     reuse=reuse)
            # positional embedding, shared for both K&V
            self.position_embedding_table = embedding(vocab_size=self.seqlen,
                                                      embedding_dim=self.embedding_dim,
                                                      zero_pad=False,
                                                      use_reg=self.use_reg,
                                                      l2_reg=self.l2_emb,
                                                      scope='position_embedding_table',
                                                      reuse=reuse)
            # attention noise
            self.sigma_noise = tf.compat.v1.Variable(0.1 * tf.ones(self.num_global_heads),
                                                     trainable=True,
                                                     name=f'sigma_noise',
                                                     dtype=tf.float32)

    ###############################################################################
    #               NET INFERENCES
    ###############################################################################
    def _create_net_inference(self, name, reuse=None):
        self.logger.debug(f'--> Create inference for {name}')
        super(Mojito, self)._create_net_inference(name, reuse=reuse)
        with tf.compat.v1.variable_scope('shared_input_comp',
                                         reuse=reuse):
            self.users = tf.nn.embedding_lookup(self.user_embedding_table,
                                                self.user_ids)
            # trans vectors
            self.fism_items = tf.nn.embedding_lookup(
                self.item_embedding_table, self.item_fism_ids)
            self.user_fism_items = tf.concat([
                tf.expand_dims(self.users, axis=1), self.fism_items],
                axis=1)
            self.nonscale_input_seq = tf.nn.embedding_lookup(
                self.item_embedding_table, self.seq_ids)

            # absolute position sequence representation
            self.abs_position = self._learnable_abs_position_embedding(
                self.position_embedding_table)
            self.seq += self.abs_position

            # context representations
            self.ctx_seq = self._ctx_representation(reuse=reuse,
                                                    year_ids=self.seq_year_ids,
                                                    month_ids=self.seq_month_ids,
                                                    day_ids=self.seq_day_ids,
                                                    dayofweek_ids=self.seq_dayofweek_ids,
                                                    dayofyear_ids=self.seq_dayofyear_ids,
                                                    week_ids=self.seq_week_ids,
                                                    hour_ids=self.seq_hour_ids,
                                                    seqlen=self.seqlen,
                                                    use_year=self.use_year,
                                                    activation=self.ctx_activation,
                                                    name='ctx_input_seq')
            ctx_seq = tf.identity(self.ctx_seq)
            if self.input_scale is True:
                self.logger.info('Scale context sequences')
                ctx_seq = ctx_seq * (self.embedding_dim ** 0.5)
            loc_ctx_seq = ctx_seq + self.abs_position
        mask = self._get_mask()
        self.loc_seq = self._seq_representation(self.seq, loc_ctx_seq,
                                                sigma_noise=self.sigma_noise,
                                                mask=mask,
                                                reuse=reuse,
                                                causality=self.causality,
                                                name='local')

    def _seq_representation(self, seq, ctx_seq, sigma_noise,
                            mask, reuse, causality, name=''):
        with tf.compat.v1.variable_scope(f'{name}_net_inference',
                                         reuse=reuse):
            concat_seq = tf.concat([seq, ctx_seq], axis=-1)
            out_seq = self._admix_sas_representation(
                seq=concat_seq,
                context_seq=self.ctx_seq,
                sigma_noise=sigma_noise,
                mask=mask,
                causality=causality,
                name=f'{name}_concat_seq',
                reuse=reuse)
        return out_seq

    def _admix_sas_representation(self, seq, context_seq,
                                  sigma_noise, mask, causality,
                                  name='', reuse=None):
        sigma_noise = tf.expand_dims(sigma_noise, axis=0)
        sigma_noise = tf.tile(sigma_noise, [self.batch_size, 1])
        seq = tf.compat.v1.layers.dropout(
            seq,
            rate=self.dropout_rate,
            training=tf.convert_to_tensor(self.is_training))
        seq *= mask
        seq = admix_multi_head_attention_blocks(seq=seq,
                                                context_seq=context_seq,
                                                num_blocks=self.num_blocks,
                                                num_heads=self.num_heads,
                                                dim_head=self.dim_head,
                                                sigma_noise=sigma_noise,
                                                dropout_rate=self.dropout_rate,
                                                mask=mask,
                                                output_dim=self.local_output_dim,
                                                causality=causality,
                                                residual_type=self.residual_type,
                                                is_training=self.is_training,
                                                reuse=reuse,
                                                name=f'{name}_mha_blocks')
        seq = normalize(seq)
        return seq

    def _ctx_representation(self, year_ids, month_ids,
                            day_ids, dayofweek_ids,
                            dayofyear_ids, week_ids, hour_ids,
                            seqlen, reuse, use_year=True,
                            activation=None,
                            name='shared_context_representation'):
        # time vectors
        seq_years = self.basis_time_encode(inputs=year_ids,
                                           time_dim=self.tempo_embedding_dim,
                                           expand_dim=self.expand_dim,
                                           scope='year',
                                           reuse=reuse)
        seq_months = self.basis_time_encode(inputs=month_ids,
                                            time_dim=self.tempo_embedding_dim,
                                            expand_dim=self.expand_dim,
                                            scope='month',
                                            reuse=reuse)
        seq_days = self.basis_time_encode(inputs=day_ids,
                                          time_dim=self.tempo_embedding_dim,
                                          expand_dim=self.expand_dim,
                                          scope='day',
                                          reuse=reuse)
        seq_dayofweeks = self.basis_time_encode(inputs=dayofweek_ids,
                                                time_dim=self.tempo_embedding_dim,
                                                expand_dim=self.expand_dim,
                                                scope='dayofweek',
                                                reuse=reuse)
        seq_dayofyears = self.basis_time_encode(inputs=dayofyear_ids,
                                                time_dim=self.tempo_embedding_dim,
                                                expand_dim=self.expand_dim,
                                                scope='dayofyear',
                                                reuse=reuse)
        seq_weeks = self.basis_time_encode(inputs=week_ids,
                                           time_dim=self.tempo_embedding_dim,
                                           expand_dim=self.expand_dim,
                                           scope='week',
                                           reuse=reuse)
        seq_hours = self.basis_time_encode(inputs=hour_ids,
                                           time_dim=self.tempo_embedding_dim,
                                           expand_dim=self.expand_dim,
                                           scope='hour',
                                           reuse=reuse)
        if use_year is True:
            ctx_seq_concat = tf.concat([seq_years, seq_months,
                                        seq_days, seq_dayofweeks,
                                        seq_dayofyears, seq_weeks, seq_hours],
                                       axis=-1)
            ctx_seq_concat = tf.reshape(
                ctx_seq_concat,
                shape=[tf.shape(self.seq_ids)[0] * seqlen,
                       self.num_contexts * self.tempo_embedding_dim])
        else:
            ctx_seq_concat = tf.concat([seq_months,
                                        seq_days, seq_dayofweeks,
                                        seq_dayofyears, seq_weeks, seq_hours], axis=-1)
            ctx_seq_concat = tf.reshape(
                ctx_seq_concat,
                shape=[tf.shape(self.seq_ids)[0] * seqlen,
                       (self.num_contexts - 1) * self.tempo_embedding_dim])

        ctx_seq = tf.compat.v1.layers.dense(
            inputs=ctx_seq_concat,
            units=self.embedding_dim,
            activation=activation,
            reuse=reuse,
            kernel_initializer=tf.random_normal_initializer(
                stddev=0.01), name=f'{name}_dense_output')
        ctx_seq = tf.compat.v1.layers.dropout(
            ctx_seq,
            rate=self.dropout_rate,
            training=tf.convert_to_tensor(self.is_training))
        ctx_seq = tf.reshape(
            ctx_seq,
            shape=[tf.shape(self.seq_ids)[0], seqlen, self.embedding_dim],
            name=f'{name}_context_embComp')
        return ctx_seq

    ###############################################################################
    #               TEST INFERENCES
    ###############################################################################
    def _create_test_inference(self, name, reuse=None):
        # lookup embedding for test items
        # shape = [bs, ntest, dim]
        test_item_emb = tf.nn.embedding_lookup(self.item_embedding_table,
                                               self.test_item_ids)
        test_ctx_seq = self._test_context_seq(reuse=tf.compat.v1.AUTO_REUSE)
        fused_test_item_emb = tf.concat([test_item_emb, test_ctx_seq],
                                        axis=-1)

        # dot product is used to measure affinity
        self.loc_test_logits = tf.matmul(self.loc_seq,
                                         tf.transpose(fused_test_item_emb, perm=[0, 2, 1]))
        if self.lambda_glob > 0:
            att_seq = self._fism_attentive_vectors(self.user_fism_items,
                                                   self.nonscale_input_seq)
            glob_seq_vecs = self.nonscale_input_seq * (1.0 - self.lambda_trans_seq) + \
                            (self.nonscale_input_seq * att_seq) * self.lambda_trans_seq
            glob_seq_vecs = tf.reduce_sum(glob_seq_vecs[:, 1:, :], axis=1,
                                          keepdims=True)
            glob_test_atts = self._fism_attentive_vectors(self.user_fism_items,
                                                          test_item_emb)
            glob_test_logits = test_item_emb * (1.0 - self.lambda_trans_seq) + \
                               (test_item_emb * glob_test_atts) * self.lambda_trans_seq
            glob_test_logits = (glob_test_logits + glob_seq_vecs) / self.seqlen
            glob_test_logits = tf.reduce_sum(glob_test_logits * test_item_emb,
                                             axis=-1)
            loc_test_logits = self.loc_test_logits[:, -1, :]
            self.test_logits = loc_test_logits + self.lambda_glob * glob_test_logits
        else:
            self.test_logits = self.loc_test_logits
            # test logits shape: [batch, seq_len, n_test_items]
            # take only the last item for the most recent interaction
            self.test_logits = self.test_logits[:, -1, :]

    def _test_context_seq(self, reuse=None):
        with tf.compat.v1.variable_scope('shared_input_comp',
                                         reuse=reuse):
            test_year_ids = tf.tile(tf.expand_dims(self.test_year_ids, axis=-1),
                                    [1, self.num_test_negatives + 1])
            test_month_ids = tf.tile(tf.expand_dims(self.test_month_ids, axis=-1),
                                     [1, self.num_test_negatives + 1])
            test_day_ids = tf.tile(tf.expand_dims(self.test_day_ids, axis=-1),
                                   [1, self.num_test_negatives + 1])
            test_dayofweek_ids = tf.tile(tf.expand_dims(self.test_dayofweek_ids, axis=-1),
                                         [1, self.num_test_negatives + 1])
            test_dayofyear_ids = tf.tile(tf.expand_dims(self.test_dayofyear_ids, axis=-1),
                                         [1, self.num_test_negatives + 1])
            test_week_ids = tf.tile(tf.expand_dims(self.test_week_ids, axis=-1),
                                    [1, self.num_test_negatives + 1])
            test_hour_ids = tf.tile(tf.expand_dims(self.test_hour_ids, axis=-1),
                                    [1, self.num_test_negatives + 1])
            test_ctx_seq = self._ctx_representation(
                year_ids=test_year_ids, month_ids=test_month_ids,
                day_ids=test_day_ids, dayofweek_ids=test_dayofweek_ids,
                dayofyear_ids=test_dayofyear_ids, week_ids=test_week_ids,
                hour_ids=test_hour_ids,
                seqlen=self.num_test_negatives + 1,
                reuse=tf.compat.v1.AUTO_REUSE,
                use_year=self.use_year,
                activation=self.ctx_activation,
                name='ctx_input_seq')
        return test_ctx_seq

    ###############################################################################
    #               LOSS
    ###############################################################################
    def _create_loss(self):
        """
        Build loss graph
        :return:
        """
        loc_loss = self._loss_on_seq(self.loc_seq)
        if self.lambda_glob > 0:
            pos_seq = tf.reshape(
                self.pos_emb,
                shape=[tf.shape(self.seq_ids)[0], self.seqlen, self.embedding_dim])
            neg_seq = tf.reshape(
                self.neg_emb,
                shape=[tf.shape(self.seq_ids)[0], self.seqlen, self.embedding_dim])
            if self.lambda_trans_seq > 0:
                pos_att_vecs = self._adaptive_attentive_seq(
                    self.pos_emb,
                    user_fism_items=self.user_fism_items,
                    need_reshaped=True,
                    name='adaptive_pos_sequence')
            else:
                pos_att_vecs = tf.reshape(
                    self.pos_emb,
                    shape=[tf.shape(self.seq_ids)[0], self.seqlen, self.embedding_dim])
            pos_logits = tf.reshape(
                tf.reduce_sum(pos_att_vecs * pos_seq, axis=-1),
                shape=[tf.shape(self.seq_ids)[0] * self.seqlen])

            if self.lambda_trans_seq > 0:
                neg_att_vecs = self._adaptive_attentive_seq(
                    self.neg_emb,
                    user_fism_items=self.user_fism_items,
                    need_reshaped=True,
                    name='adaptive_neg_sequence')
            else:
                neg_att_vecs = tf.reshape(
                    self.neg_emb,
                    shape=[tf.shape(self.seq_ids)[0], self.seqlen, self.embedding_dim])
            neg_logits = tf.reshape(
                tf.reduce_sum(neg_att_vecs * neg_seq, axis=-1),
                shape=[tf.shape(self.seq_ids)[0] * self.seqlen])
            # Regularization
            l2_norm = tf.add_n([
                self.lambda_user * tf.reduce_sum(tf.multiply(
                    self.user_embedding_table, self.user_embedding_table)),
                self.lambda_item * tf.reduce_sum(tf.multiply(
                    self.item_embedding_table, self.item_embedding_table))
            ])
            glob_loss = tf.reduce_sum(
                - tf.math.log(tf.sigmoid(pos_logits) + 1e-24) * self.istarget -
                tf.math.log(1 - tf.sigmoid(neg_logits) + 1e-24) * self.istarget
            ) / tf.reduce_sum(self.istarget) + l2_norm
            self.loss = loc_loss + self.lambda_glob * glob_loss
        else:
            self.loss = loc_loss
        self.reg_loss = sum(tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES))
        self.loss += self.reg_loss

    def _loss_on_seq(self, seq):
        with tf.compat.v1.variable_scope('shared_input_comp',
                                         reuse=tf.compat.v1.AUTO_REUSE):
            seq_emb = tf.reshape(
                seq,
                shape=[tf.shape(self.seq_ids)[0] * self.seqlen, self.local_output_dim])
            pos_ctx_seq = self._ctx_representation(
                year_ids=self.pos_year_ids, month_ids=self.pos_month_ids,
                day_ids=self.pos_day_ids, dayofweek_ids=self.pos_dayofweek_ids,
                dayofyear_ids=self.pos_dayofyear_ids, week_ids=self.pos_week_ids,
                hour_ids=self.pos_hour_ids,
                seqlen=self.seqlen,
                reuse=tf.compat.v1.AUTO_REUSE,
                use_year=self.use_year,
                activation=self.ctx_activation,
                name='ctx_input_seq')
            ctx_emb = tf.reshape(
                pos_ctx_seq,
                shape=[tf.shape(self.seq_ids)[0] * self.seqlen, self.embedding_dim])
        pos_emb = tf.concat([self.pos_emb, ctx_emb], axis=-1)
        neg_emb = tf.concat([self.neg_emb, ctx_emb], axis=-1)
        # prediction layer
        pos_logits = tf.reduce_sum(pos_emb * seq_emb, -1)
        neg_logits = tf.reduce_sum(neg_emb * seq_emb, -1)
        loss = tf.reduce_sum(
            - tf.math.log(tf.sigmoid(pos_logits) + 1e-24) * self.istarget -
            tf.math.log(1 - tf.sigmoid(neg_logits) + 1e-24) * self.istarget
        ) / tf.reduce_sum(self.istarget)
        return loss

    ###############################################################################
    #               MISC
    ###############################################################################
    def basis_time_encode(self, inputs, time_dim, expand_dim,
                          scope='basis_time_kernel', reuse=None,
                          return_weight=False):
        """Mercer's time encoding

        Args:
          inputs: A 2d float32 tensor with shate of [N, max_len]
          time_dim: integer, number of dimention for time embedding
          expand_dim: degree of frequency expansion
          scope: string, scope for tensorflow variables
          reuse: bool, if true the layer could be reused
          return_weight: bool, if true return both embeddings and frequency

        Returns:
          A 3d float tensor which embeds the input or
          A tuple with one 3d float tensor (embeddings) and 2d float tensor (frequency)
        """

        # inputs: [N, max_len]
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            expand_input = tf.tile(tf.expand_dims(inputs, 2),
                                   [1, 1, time_dim])  # [N, max_len, time_dim]
            init_period_base = np.linspace(0, self.tempo_linspace, time_dim)
            init_period_base = init_period_base.astype(np.float32)
            period_var = tf.compat.v1.get_variable('time_cos_freq',
                                                   dtype=tf.float32,
                                                   initializer=tf.constant(init_period_base))
            period_var = 10.0 ** period_var
            period_var = tf.tile(tf.expand_dims(period_var, 1),
                                 [1, expand_dim])  # [time_dim] -> [time_dim, 1] -> [time_dim, expand_dim]
            expand_coef = tf.cast(tf.reshape(tf.range(expand_dim) + 1, [1, -1]), tf.float32)

            freq_var = 1 / period_var
            freq_var = freq_var * expand_coef

            basis_expan_var = tf.compat.v1.get_variable(
                'basis_expan_var',
                shape=[time_dim, 2 * expand_dim],
                initializer=tf.compat.v1.glorot_uniform_initializer())

            basis_expan_var_bias = tf.compat.v1.get_variable(
                'basis_expan_var_bias',
                shape=[time_dim],
                initializer=tf.zeros_initializer)  # initializer=tf.glorot_uniform_initializer())

            sin_enc = tf.sin(tf.multiply(tf.expand_dims(expand_input, -1),
                                         tf.expand_dims(tf.expand_dims(freq_var, 0), 0)))
            cos_enc = tf.cos(tf.multiply(tf.expand_dims(expand_input, -1),
                                         tf.expand_dims(tf.expand_dims(freq_var, 0), 0)))
            time_enc = tf.multiply(tf.concat([sin_enc, cos_enc], axis=-1),
                                   tf.expand_dims(tf.expand_dims(basis_expan_var, 0), 0))
            time_enc = tf.add(tf.reduce_sum(time_enc, -1),
                              tf.expand_dims(tf.expand_dims(basis_expan_var_bias, 0), 0))

        if return_weight:
            return time_enc, freq_var
        return time_enc

    def _adaptive_attentive_seq(self, seq, user_fism_items,
                                name='', need_reshaped=True):
        if need_reshaped is True:
            seq = tf.reshape(
                seq,
                shape=[tf.shape(self.seq_ids)[0], self.seqlen, self.embedding_dim])
        att_seq = self._fism_attentive_vectors(user_fism_items, seq,
                                               name=name)
        if self.lambda_trans_seq < 1:
            att_fism_seq = seq * (1.0 - self.lambda_trans_seq) + \
                           (seq * att_seq) * self.lambda_trans_seq
        else:
            att_fism_seq = seq * att_seq
        return att_fism_seq

    def _fism_attentive_vectors(self, fism_items, seq, name=''):
        with tf.name_scope(name):
            w_ij = tf.matmul(seq,
                             tf.transpose(fism_items, perm=[0, 2, 1]))
            exp_wij = tf.exp(w_ij)
            exp_sum = tf.reduce_sum(exp_wij, axis=-1, keepdims=True)
            if self.beta != 1.0:
                exp_sum = tf.pow(exp_sum,
                                 tf.constant(self.beta, tf.float32, [1]))
            att = exp_wij / exp_sum
            att_vecs = tf.matmul(att, fism_items)
        return att_vecs
