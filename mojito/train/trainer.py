import os
import time
import numpy as np
from tqdm import tqdm

from mojito.logging import get_logger
from mojito.data.loaders import dataloader_factory
from mojito.eval.evaluator import Evaluator


class Trainer:
    """
    Trainer is responsible to estimate paramaters
    for a given model
    """
    def __init__(self, sess, model, params):
        """
        Initialization a trainer. The trainer will be responsible
        to train the model
        :param sess: global session
        :param model: model to be trained
        :param params: hyperparameters for training
        """
        self.sess = sess
        self.model = model
        self.params = params
        self.model_dir = params['training']['model_dir']
        self.n_epochs = self.params['training'].get('num_epochs', 20)
        self.n_valid_users = self.params['training'].get('n_valid_users',
                                                         10000)
        self.num_negtives = self.params['training'].get(
            'num_negatives', 100)
        self.logger = get_logger()

    def fit(self, data):
        """
        Training model
        :param data:
        :return:
        """
        # create data loaders for train & validation
        cache_path = self.params['cache']['path']
        dataset_params = self.params['dataset']
        u_ncore = dataset_params.get('u_ncore', 1)
        i_ncore = dataset_params.get('i_ncore', 1)
        item_type = dataset_params.get('item_type', 'item')
        spec = f'{item_type}_{u_ncore}core-users_' \
               f'{i_ncore}core-items'
        training_params = self.params['training']
        model_params = training_params['model']
        random_seed = self.params['dataset'].get('random_state', 2022)
        model_name = model_params['name']
        timespan = model_params['params'].get(
            'timespan', 256)
        n_fism_items = -1
        fism_sampling = 'uniform'
        fism_type = 'item'
        fism_beta = 1.0
        if 'fism' in model_params['params']:
            n_fism_items = model_params['params']['fism']['n_items']
            fism_sampling = model_params['params']['fism']['sampling']
            fism_type = model_params['params']['fism']['type']
            fism_beta = model_params['params']['fism']['beta']
        metrics_path = '{}/metrics.csv'.format(self.model_dir)
        if os.path.isfile(metrics_path):
            os.remove(metrics_path)

        active_sse = True if 'sse' in model_params['params'] and \
                             model_params['params']['sse']['activate'] is True \
            else False
        sse_type = 'uniform'
        threshold_item = threshold_favs = threshold_user = 1.0
        if active_sse is True:
            sse_type = model_params['params']['sse']['type']
            threshold_item = model_params['params']['sse']['threshold_item']
            if 'threshold_favs' in model_params['params']['sse']:
                threshold_favs = model_params['params']['sse']['threshold_favs']
            if 'threshold_user' in model_params['params']['sse']:
                threshold_user = model_params['params']['sse']['threshold_user']

        neg_sampling = 'uniform'
        if 'negative_sampling' in self.params['eval']:
            neg_sampling = self.params['eval']['negative_sampling']['type']
        best_valid_score = -1.0
        best_ep = -1
        seqlen = model_params['params'].get('seqlen', 50)
        with open(metrics_path, 'w') as f:
            header = 'epoch,lr,train_loss,val_loss,ndcg,hr,' \
                     'rep_ndcg,rep_hr,exp_ndcg,exp_hr'
            f.write(f'{header}\n')
            # for each epoch
            for ep in range(self.n_epochs):
                start_time = time.time()
                train_dataloader = dataloader_factory(
                    data=data,
                    batch_size=training_params['batch_size'],
                    seqlen=seqlen,
                    mode='train',
                    random_seed=random_seed,
                    cache_path=cache_path,
                    spec=spec,
                    epoch=ep,
                    model_name=model_name,
                    timespan=timespan,
                    activate_sse=active_sse,
                    sse_type=sse_type,
                    threshold_item=threshold_item,
                    threshold_favs=threshold_favs,
                    threshold_user=threshold_user,
                    n_fism_items=n_fism_items,
                    fism_sampling=fism_sampling,
                    fism_type=fism_type,
                    fism_beta=fism_beta,
                    train_interaction_indexes=data['train_interaction_indexes'])
                # calculate train loss
                train_loss, train_reg_loss = self._get_epoch_loss(
                    train_dataloader, ep)
                self.logger.info(f'Train loss: {train_loss}, reg_loss: {train_reg_loss}')
                valid_batchsize = training_params['batch_size']
                valid_dataloader = dataloader_factory(
                    data=data,
                    batch_size=valid_batchsize,
                    seqlen=seqlen,
                    mode='valid',
                    random_seed=random_seed,
                    cache_path=cache_path,
                    model_name=model_name,
                    num_scored_users=data['num_valid_users'],
                    timespan=timespan,
                    n_fism_items=n_fism_items,
                    fism_sampling=fism_sampling,
                    fism_type=fism_type,
                    fism_beta=fism_beta,
                    neg_sampling=neg_sampling,
                    epoch=ep)
                # get predictions on valid_set
                score = Evaluator.eval(valid_dataloader, self.model)
                if best_valid_score < score[0] or ep == 1:
                    save_model = True
                    best_valid_score = score[0]
                    best_ep = ep
                else:
                    save_model = False
                logged_message = self._get_message(
                    ep, self.model.learning_rate,
                    train_loss, score, start_time)
                self.logger.info(', '.join(logged_message))
                metric_message = self._get_message(
                    ep, self.model.learning_rate,
                    train_loss, score, start_time,
                    logged=False)
                f.write(','.join(metric_message) + '\n')
                f.flush()
                if save_model:
                    save_path = f'{self.model_dir}/' \
                                f'{self.model.__class__.__name__.lower()}' \
                                f'-epoch_{ep}'
                    self.model.save(save_path=save_path, global_step=ep)
            self.logger.info(f'Best validation : {best_valid_score}, '
                             f'on epoch {best_ep}')

    def _get_epoch_loss(self, dataloader, epoch_id):
        """
        Forward pass for an epoch
        :param dataloader:
        :param epoch_id:
        :return:
        """
        n_batches = dataloader.get_num_batches()
        losses = []
        reg_losses = []
        desc = f'Optimizing epoch #{epoch_id}'
        # for each batch
        for _ in tqdm(range(1, n_batches), desc=f'{desc}...'):
            # get batch data
            batch_data = dataloader.next_batch()
            batch_loss, batch_reg_loss = self._get_batch_loss(batch=batch_data)
            if type(batch_loss) == np.ndarray:
                batch_loss = np.mean(batch_loss)
            if not np.isinf(batch_loss) and not np.isnan(batch_loss):
                losses.append(batch_loss)
            if not np.isinf(batch_reg_loss) and not np.isnan(batch_reg_loss):
                reg_losses.append(batch_reg_loss)
        loss = np.mean(losses, axis=0)
        reg_loss = np.mean(reg_losses, axis=0)
        return loss, reg_loss

    def _get_batch_loss(self, batch):
        """
        Forward pass for a batch
        :param batch:
        :return:
        """
        feed_dict = self.model.build_feedict(batch, is_training=True)
        reg_loss = 0.0
        _, loss = self.sess.run(
            [self.model.train_ops, self.model.loss],
            feed_dict=feed_dict)
        return loss, reg_loss

    @classmethod
    def _get_message(cls, ep, learning_rate,
                     train_loss, score, start_time, logged=True):
        duration = int(time.time() - start_time)
        ss, duration = duration % 60, duration // 60
        mm, hh = duration % 60, duration // 60
        if logged is True:
            message = [f'Epoch #{ep}',
                       f'LR {learning_rate:6.5f}',
                       f'Tr-Loss {train_loss:7.5f}',
                       f'Val NDCG {score[0]:7.5f}',
                       f'Val HR {score[1]:7.5f}',
                       f'Dur:{hh:0>2d}h{mm:0>2d}m{ss:0>2d}s']
        else:
            message = [f'{ep}:',
                       f'{learning_rate:6.7f}',
                       f'{train_loss:7.5f}',
                       f'{score[0]:7.5f}',
                       f'{score[1]:7.5f}']
        return message
