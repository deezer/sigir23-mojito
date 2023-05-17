import os
import tensorflow as tf

from mojito.logging import get_logger
from mojito.utils.params import process_params
from mojito.data.datasets import dataset_factory
from mojito.models import ModelFactory
from mojito.train.trainer import Trainer


def entrypoint(params):
    """ Command entrypoint
    :param params: Deserialized JSON configuration file
                   provided in CLI args.
    """
    logger = get_logger()
    # process params
    training_params, model_params = process_params(params)

    # create model directory if not exist
    if not os.path.exists(training_params['model_dir']):
        os.makedirs(training_params['model_dir'], exist_ok=True)
    logger.info(training_params['model_dir'])

    # load datasets
    data = dataset_factory(params=params)
    # start model training
    sess_config = tf.compat.v1.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess_config.allow_soft_placement = True
    with tf.compat.v1.Session(config=sess_config) as sess:
        # create model
        model = ModelFactory.generate_model(sess=sess,
                                            params=training_params,
                                            n_users=data['n_users'],
                                            n_items=data['n_items'])
        sess.run(tf.compat.v1.global_variables_initializer())

        # create a trainer to train model
        trainer = Trainer(sess, model, params)

        logger.info('Start model training')
        trainer.fit(data=data)
        logger.info('Model training done')
