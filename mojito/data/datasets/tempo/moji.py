import os
import pickle
import numpy as np
from tqdm import tqdm

from mojito.data.datasets.tempo import TempoDataset
from mojito.utils.tempo import ts_to_high_level_tempo


class MojitoDataset(TempoDataset):
    def __init__(self, params):
        super(MojitoDataset, self).__init__(params)
        self._gen_fism()

    def _fetch_tempo_data(self):
        self.logger.debug('Fetch additional temporal data')
        tempo_data_path = os.path.join(self.cache_path,
                                       self.model_name)
        if not os.path.exists(tempo_data_path):
            os.makedirs(tempo_data_path)
        timedict_path = os.path.join(tempo_data_path, 'timedict.pkl')
        if not os.path.exists(timedict_path):
            train_set = self.data['train_set']
            valid_set = self.data['valid_set']
            test_set = self.data['test_set']
            time_set = self._time_set(
                train_set, indexes=self.data['train_interaction_indexes'])
            time_set = time_set.union(self._time_set(valid_set),
                                      self._time_set(test_set))
            time_dict = ts_to_high_level_tempo(time_set=time_set)
            pickle.dump(time_dict, open(timedict_path, 'wb'))
        else:
            time_dict = pickle.load(open(timedict_path, 'rb'))
        self.data['time_dict'] = time_dict
        self.logger.debug('Finish fetch additional temporal data')

    def _gen_fism(self, n_epochs=100):
        n_items = self.model_params['fism']['n_items']
        if n_items > 0:
            self.logger.debug(f'Fetch additional representative items for FISM')
            self._gen_fism_type('item', n_epochs=n_epochs, n_items=n_items)

    def _gen_fism_type(self, fism_type, n_epochs, n_items):
        fism_sampling = self.model_params['fism']['sampling']
        fism_beta = self.model_params['fism']['beta']
        train_interactions_prefix = self.cache_params['train_interactions']
        tempo_data_path = os.path.join(self.cache_path,
                                       'fism', fism_type,
                                       f'nitems{n_items}',
                                       f'{fism_sampling}-sampling')
        if not os.path.exists(tempo_data_path):
            os.makedirs(tempo_data_path)
        prefix = 'item_then_user' if fism_type == 'user' \
            else 'user_then_item'
        elem_freq_path = os.path.join(
            self.cache_path,
            f'samples-step{self.dataset_params["samples_step"]}_'
            f'{train_interactions_prefix}_aggregated_{prefix}_dict_{self.common}_'
            f'seqlen{self.seqlen}.pkl')
        elem_freq_dict = pickle.load(open(elem_freq_path, 'rb'))
        for ep in range(n_epochs):
            if fism_sampling != 'uniform':
                fism_name = f'fism_nitems{n_items}_' \
                            f'seqlen{self.seqlen}_' \
                            f'{fism_sampling}-sampling_' \
                            f'beta{fism_beta}_' \
                            f'ep{ep}.pkl'
            else:
                fism_name = f'fism_nitems{n_items}_' \
                            f'seqlen{self.seqlen}_' \
                            f'{fism_sampling}-sampling_' \
                            f'ep{ep}.pkl'
            fism_path = os.path.join(
                tempo_data_path, fism_name)
            if not os.path.exists(fism_path):
                fism_dict = {}
                n_elems = self.data['n_items'] if fism_type == 'user' \
                    else self.data['n_users']
                start_idx = 1
                end_idx = n_elems + 1
                for eid in tqdm(range(start_idx, end_idx),
                                desc='Generating...'):
                    elem_dict = elem_freq_dict[eid]
                    iid_freq_list = [(k, v) for k, v in elem_dict.items()]
                    iids, freqs = zip(*iid_freq_list)
                    freqs = np.array(freqs, dtype=np.float32)
                    if fism_beta != 1.0:
                        freqs = freqs ** fism_beta
                    freqs = freqs / np.sum(freqs)
                    if n_items >= len(iids):
                        chosen_items = iids
                    else:
                        chosen_indexes = []
                        if fism_sampling == 'uniform':
                            chosen_indexes = np.random.permutation(
                                np.arange(len(iids)))[:n_items]
                        elif fism_sampling == 'pop':
                            chosen_indexes = np.random.choice(np.arange(len(iids)),
                                                              size=n_items,
                                                              p=freqs,
                                                              replace=False)
                        chosen_items = [iids[idx] for idx in chosen_indexes]
                    fism_dict[eid] = chosen_items
                pickle.dump(fism_dict, open(fism_path, 'wb'))

