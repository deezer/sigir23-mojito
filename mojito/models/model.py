import tensorflow as tf

from mojito import MojitoError
from mojito.logging import get_logger


class Model:
    """
    Abstract model
    """
    # Supported optimizers.
    ADADELTA = 'Adadelta'
    SGD = 'SGD'
    ADAM = 'Adam'

    def __init__(self, sess, params, n_users, n_items):
        """
        Initialize a model
        :param sess: global session
        :param params: model parameters
        :param n_users: number of users
        :param n_items: number of items
        """
        self.logger = get_logger()
        self.sess = sess
        self.learning_rate = params.get('learning_rate', 0.001)
        self.embedding_dim = params.get('embedding_dim', 32)
        self.use_reg = params['model']['params'].get('use_reg', True)
        self.l2_emb = params['model']['params'].get('l2_emb', 0.0)
        self.model_dir = params.get('model_dir', 'exp/model')
        self.n_epochs = params.get('n_epochs', 20)
        self.seqlen = params['model']['params'].get('seqlen', 50)
        self.batch_size = params.get('batch_size', 256)
        self.n_users = n_users
        self.n_items = n_items
        self.optimizer = params['optimizer']
        self.checkpoint = None
        self.saver = None
        self.max_to_keep = params.get('max_to_keep', 1)
        self.dropout_rate = params['model']['params'].get(
            'dropout_rate', 0.2)
        self.num_test_negatives = params.get('num_test_negatives', 100)
        # loss
        self.loss = None
        # prediction
        self.test_logits = None

    def build_graph(self, name=None):
        """
        Build model computation graph
        :return:
        """
        self._create_placeholders()
        self._create_variables(reuse=tf.compat.v1.AUTO_REUSE)
        self._create_inference(name, reuse=tf.compat.v1.AUTO_REUSE)
        self._create_test_inference(name, reuse=tf.compat.v1.AUTO_REUSE)
        self._create_loss()
        self._create_train_ops()
        self.saver = tf.compat.v1.train.Saver(
            max_to_keep=self.max_to_keep)

    def save(self, save_path, global_step):
        """
        Save the model to directory
        :param save_path:
        :param global_step:
        :return:
        """
        self.saver.save(self.sess, save_path=save_path,
                        global_step=global_step)

    def restore(self, name=None):
        """
        Restore the model if it already exists
        :return:
        """
        self.checkpoint = tf.compat.v1.train.get_checkpoint_state(
            self.model_dir)
        if self.checkpoint is not None:
            self.logger.info(f'Load {self.__class__} model '
                             f'from {self.model_dir}')
            self.build_graph(name=name)
            self.saver.restore(self.sess,
                               self.checkpoint.model_checkpoint_path)

    def build_feedict(self, batch, is_training=True):
        raise NotImplementedError('build_feedict method should be '
                                  'implemented in concrete model')

    def predict(self, feed_dict):
        return self.sess.run(self.test_logits, feed_dict)

    def export_embeddings(self):
        raise NotImplementedError('export_embeddings method should be '
                                  'implemented in concrete model')

    def _create_placeholders(self):
        """
        Build input graph
        :return:
        """
        self.logger.debug('--> Create input placeholders')
        with tf.name_scope('input_data'):
            # boolean to check if training, used for dropout
            self.is_training = tf.compat.v1.placeholder(
                name='is_training',
                dtype=tf.bool,
                shape=())
            # batch of user ids
            self.user_ids = tf.compat.v1.placeholder(name='user_ids',
                                                     dtype=tf.int32,
                                                     shape=[None])
            # for each positive item in test, concat num_negatives items
            # then use predicted outputs to evaluate the model quality
            self.test_item_ids = tf.compat.v1.placeholder(
                name='test_item_ids', dtype=tf.int32,
                shape=[None, self.num_test_negatives + 1])

    def _create_variables(self, reuse=None):
        """
        Build variables
        :return:
        """
        raise NotImplementedError('_create_variables method should be '
                                  'implemented in concrete model')

    def _create_inference(self, name, reuse=None):
        """
        Build inference graph
        :return:
        """
        raise NotImplementedError('_create_inference method should be '
                                  'implemented in concrete model')

    def _create_test_inference(self, name, reuse=None):
        """
        Build inference graph
        :return:
        """
        raise NotImplementedError('_create_test_inference method should be '
                                  'implemented in concrete model')

    def _create_loss(self):
        """
        Build loss graph
        :return:
        """
        raise NotImplementedError('_create_loss method should be '
                                  'implemented in concrete model')

    def _create_train_ops(self):
        """
        Train operations
        :return:
        """
        self.logger.debug('--> Define training operators')
        optimizer = self._build_optimizer(self.learning_rate)
        self.train_ops = [optimizer.minimize(self.loss)]

    def _build_optimizer(self, lr):
        """ Builds an optimizer instance from internal parameter values.
        Default to AdamOptimizer if not specified.

        :returns: Optimizer instance from internal configuration.
        """
        self.logger.debug('----> Define optimizer')
        if self.optimizer == self.ADADELTA:
            return tf.compat.v1.train.AdadeltaOptimizer()
        if self.optimizer == self.SGD:
            return tf.compat.v1.train.GradientDescentOptimizer(lr)
        elif self.optimizer == self.ADAM:
            return tf.compat.v1.train.AdamOptimizer(lr, beta2=0.98)
        else:
            raise MojitoError(f'Unknown optimizer type {self.optimizer}')

    @classmethod
    def _activation(cls, act_type='None'):
        if act_type == 'none':
            return None
        elif act_type == 'relu':
            return tf.nn.relu
        elif act_type == 'leaky_relu':
            return tf.nn.leaky_relu
        else:
            raise MojitoError(f'Not support activation of type {act_type}')
