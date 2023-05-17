import os
import pickle
import numpy as np

from mojito.logging import get_logger


class DataLoader:
    """
    DataLoader is responsible for train/valid
    batch data generation
    """
    def __init__(self, dataset, n_users, n_items,
                 batch_size, seqlen, random_seed=2022,
                 **kwargs):
        """
        Initialization
        :param dataset:
        :param n_users:
        :param n_items:
        :param batch_size:
        :param seqlen:
        :param random_seed:
        :param kwargs:
        """
        self.dataset = dataset
        self.n_users = n_users
        self.n_items = n_items
        self.batch_size = batch_size
        self.seqlen = seqlen
        self.random_seed = random_seed
        self.kwargs = kwargs
        self.kwargs['n_users'] = n_users
        self.rng = np.random.RandomState(self.random_seed)
        self.logger = get_logger()
        self.current_batch_idx = 0
        self.n_batches = 0
        self.model_name = kwargs['model_name']
        if 'n_fism_items' in self.kwargs:
            if 'user' in self.kwargs['fism_type']:
                self.kwargs['user_fism'] = self._get_fism(
                    data_path=os.path.join(self.kwargs['cache_path'],
                                           'fism', 'user'),
                    n_items=self.kwargs['n_fism_items'],
                    sampling=self.kwargs['fism_sampling'],
                    ep=self.kwargs['epoch'])
            if 'item' in self.kwargs['fism_type']:
                self.kwargs['item_fism'] = self._get_fism(
                    data_path=os.path.join(self.kwargs['cache_path'],
                                           'fism', 'item',
                                           f'nitems{self.kwargs["n_fism_items"]}',
                                           f'{self.kwargs["fism_sampling"]}-sampling'),
                    n_items=self.kwargs['n_fism_items'],
                    sampling=self.kwargs['fism_sampling'],
                    beta=self.kwargs['fism_beta'],
                    ep=self.kwargs['epoch'])

    def next_batch(self):
        if self.current_batch_idx == self.n_batches:
            self.current_batch_idx = 0
        batch_samples = self._batch_sampling(self.current_batch_idx)
        self.current_batch_idx += 1
        return batch_samples

    def get_num_batches(self):
        return self.n_batches

    def _batch_sampling(self, batch_index):
        """
        Batch sampling
        :param batch_index:
        :return:
        """
        raise NotImplementedError('process method should be '
                                  'implemented in concrete model')

    def _get_time_matrix(self, data_path, mode, spec):
        fin_path = os.path.join(
            data_path,
            f'{mode}_time_relation_matrix_{spec}.pkl')
        self.logger.info(f'Load time matrix from {fin_path}')
        time_matrix = pickle.load(open(fin_path, 'rb'))
        return time_matrix

    def _get_fism(self, data_path, n_items, sampling, ep, beta=1.0):
        if sampling != 'uniform':
            fism_name = f'fism_nitems{n_items}_' \
                        f'seqlen{self.seqlen}_' \
                        f'{sampling}-sampling_' \
                        f'beta{beta}_' \
                        f'ep{ep}.pkl'
        else:
            fism_name = f'fism_nitems{n_items}_' \
                        f'seqlen{self.seqlen}_' \
                        f'{sampling}-sampling_' \
                        f'ep{ep}.pkl'
        fism_path = os.path.join(data_path, fism_name)
        self.logger.info(f'Load FISM items from {fism_path}')
        fism_items = pickle.load(open(fism_path, 'rb'))
        return fism_items
