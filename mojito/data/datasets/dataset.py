import os
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict

from mojito.logging import get_logger
from mojito.data.splitter import split_data
from mojito.utils import float_dd


class Dataset:
    """
    Dataset
    """
    def __init__(self, params):
        cache_params = params['cache']
        self.dataset_params = params['dataset']
        self.eval_params = params['eval']
        self.neg_beta = 1.0
        if 'negative_sampling' in self.eval_params:
            self.neg_beta = self.eval_params['negative_sampling']['beta']
        dataset_name = self.dataset_params['name']
        self.model_name = params['training']['model']['name']
        self.u_ncore = self.dataset_params.get('u_ncore', 1)
        self.i_ncore = self.dataset_params.get('i_ncore', 1)
        self.item_type = self.dataset_params.get('item_type',
                                                 'item')
        self.activate_repeat_consumption = self.dataset_params.get('repeat', False)
        self.cache_path = cache_params['path']
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)
        self.seqlen = params['training']['model']['params'].get(
            'seqlen', 50)
        self.num_test_negatives = params['training'].get(
            'num_test_negatives', 100)
        self.num_valid_users = params['training'].get(
            'num_valid_users', -1)
        train_interactions_prefix = cache_params['train_interactions']
        valid_interactions_prefix = cache_params['valid_interactions']
        test_interactions_prefix = cache_params['test_interactions']
        self.samples_step = self.dataset_params.get('samples_step', 0)
        if dataset_name == 'lfm1b':
            self.frac = self.dataset_params.get('fraction')
            self.common = f'{self.item_type}_{self.u_ncore}core-users_' \
                          f'{self.i_ncore}core-items_' \
                          f'loop-{self.dataset_params["loopback_interval_in_hours"]}h_' \
                          f'hist-{self.dataset_params["ndays_min"]}days'
            if self.frac < 1.0:
                self.train_interactions_path = os.path.join(
                    self.cache_path, f'frac{self.frac}_{train_interactions_prefix}_{self.common}.pkl')
                self.valid_interactions_path = os.path.join(
                    self.cache_path, f'frac{self.frac}_{valid_interactions_prefix}_{self.common}.pkl')
                self.test_interactions_path = os.path.join(
                    self.cache_path, f'frac{self.frac}_{test_interactions_prefix}_{self.common}.pkl')
                self.kcore_data_name = f'frac{self.frac}_mapped_internal-ids_interactions_' \
                                       f'{self.common}.csv'
                entities_name = f'frac{self.frac}_entities_{self.common}.npz'
                self.train_interaction_indexes_path = os.path.join(
                    self.cache_path,
                    f'frac{self.frac}_train_interactions_indexes_{self.common}.pkl')
                self.train_aggregated_user_key_interaction_path = os.path.join(
                    self.cache_path,
                    f'frac{self.frac}_samples-step{self.samples_step}_'
                    f'{train_interactions_prefix}_aggregated_user_then_item_dict_{self.common}_'
                    f'seqlen{self.seqlen}.pkl')
                self.train_aggregated_item_key_interaction_path = os.path.join(
                    self.cache_path,
                    f'frac{self.frac}_samples-step{self.samples_step}_'
                    f'{train_interactions_prefix}_aggregated_item_then_user_dict_{self.common}_'
                    f'seqlen{self.seqlen}.pkl')
                self.train_aggregated_firsttime_path = os.path.join(
                    self.cache_path,
                    f'frac{self.frac}_samples-step{self.samples_step}_'
                    f'{train_interactions_prefix}_aggregated_firsttime_dict_{self.common}_'
                    f'seqlen{self.seqlen}.pkl')
            else:
                self.train_interactions_path = os.path.join(
                    self.cache_path, f'{train_interactions_prefix}_{self.common}.pkl')
                self.valid_interactions_path = os.path.join(
                    self.cache_path, f'{valid_interactions_prefix}_{self.common}.pkl')
                self.test_interactions_path = os.path.join(
                    self.cache_path, f'{test_interactions_prefix}_{self.common}.pkl')
                self.kcore_data_name = f'mapped_internal-ids_interactions_{self.common}.csv'
                entities_name = f'entities_{self.common}.npz'
                self.train_interaction_indexes_path = os.path.join(
                    self.cache_path,
                    f'train_interactions_indexes_{self.common}.pkl')
                self.train_aggregated_user_key_interaction_path = os.path.join(
                    self.cache_path,
                    f'samples-step{self.samples_step}_'
                    f'{train_interactions_prefix}_aggregated_user_then_item_dict_{self.common}_'
                    f'seqlen{self.seqlen}.pkl')
                self.train_aggregated_item_key_interaction_path = os.path.join(
                    self.cache_path,
                    f'samples-step{self.samples_step}_'
                    f'{train_interactions_prefix}_aggregated_item_then_user_dict_{self.common}_'
                    f'seqlen{self.seqlen}.pkl')
                self.train_aggregated_firsttime_path = os.path.join(
                    self.cache_path,
                    f'samples-step{self.samples_step}_'
                    f'{train_interactions_prefix}_aggregated_firsttime_dict_{self.common}_'
                    f'seqlen{self.seqlen}.pkl')
            if self.activate_repeat_consumption is not True:
                self.kcore_data_name = f'nonrepeat_{self.kcore_data_name}'
                entities_name = f'nonrepeat_{entities_name}'
            self.kcore_data_path = os.path.join(self.dataset_params['path'],
                                                self.kcore_data_name)
            self.entities_path = os.path.join(self.dataset_params['path'],
                                              entities_name)
            self.train_item_pop_path = os.path.join(
                self.cache_path,
                f'train_item_popularities_{self.common}_beta{self.neg_beta}.npz')
        else:
            self.common = f'{self.u_ncore}core-users_{self.i_ncore}core-items'
            self.train_interactions_path = os.path.join(
                self.cache_path, f'{train_interactions_prefix}_{self.common}.pkl')
            self.valid_interactions_path = os.path.join(
                self.cache_path, f'{valid_interactions_prefix}_{self.common}.pkl')
            self.test_interactions_path = os.path.join(
                self.cache_path, f'{test_interactions_prefix}_{self.common}.pkl')
            self.kcore_data_path = os.path.join(
                self.dataset_params['path'],
                f'{dataset_name}_mapped_internal-ids_interactions_{self.common}.csv')

            self.entities_path = os.path.join(self.dataset_params['path'],
                                              f'{dataset_name}_entities_{self.common}.npz')
            self.train_interaction_indexes_path = os.path.join(
                self.cache_path,
                f'train_interactions_indexes_{self.common}_samples-step{self.samples_step}.pkl')
            self.train_item_pop_path = os.path.join(
                self.cache_path,
                f'train_item_popularities_{self.common}_beta{self.neg_beta}.npz')
            self.train_aggregated_user_key_interaction_path = os.path.join(
                self.cache_path,
                f'samples-step{self.samples_step}_'
                f'{train_interactions_prefix}_aggregated_user_then_item_dict_{self.common}_'
                f'seqlen{self.seqlen}.pkl')
            self.train_aggregated_item_key_interaction_path = os.path.join(
                self.cache_path,
                f'samples-step{self.samples_step}_'
                f'{train_interactions_prefix}_aggregated_item_then_user_dict_{self.common}_'
                f'seqlen{self.seqlen}.pkl')
            self.train_aggregated_firsttime_path = os.path.join(
                self.cache_path,
                f'samples-step{self.samples_step}_'
                f'{train_interactions_prefix}_aggregated_firsttime_dict_{self.common}_'
                f'seqlen{self.seqlen}.pkl')
        self.n_epochs = params['training'].get('num_epochs', 100)
        self.logger = get_logger()
        # fetch data
        self._fetch_data()

    def _fetch_data(self):
        """
        Fetch data
        :return: data dictionary
        """
        if not os.path.exists(self.train_interactions_path) or not \
                os.path.exists(self.valid_interactions_path) or not \
                os.path.exists(self.test_interactions_path) or not \
                os.path.exists(self.entities_path):
            if not os.path.exists(self.kcore_data_path):
                # 1. read original k-core
                data = self._fetch_original_kcore_interactions()
                # 2. map original data to internal ids
                self.logger.debug(f'Map to internal ids')
                data, user_ids, item_ids = self._map_internal_ids(data)
                data.to_csv(self.kcore_data_path, sep=',',
                            header=True, index=False)
                np.savez(self.entities_path, user_ids=user_ids,
                         item_ids=item_ids)
            else:
                self.logger.debug(f'Load data from {self.kcore_data_path}')
                data = pd.read_csv(self.kcore_data_path)
                entities = np.load(self.entities_path, allow_pickle=True)
                user_ids = entities['user_ids']
                item_ids = entities['item_ids']
            # 3. sort each user interactions by timestamps
            self.logger.debug('Sort user interactions by timestamps')
            sorted_user_interactions = dict()
            grouped_data = data.groupby('user')
            for uid, interactions in grouped_data:
                interactions = interactions.sort_values(by=['timestamp'])
                iids = interactions['item'].tolist()
                times = interactions['timestamp'].tolist()
                sorted_user_interactions[uid] = list(zip(iids, times))
            # 4. split data
            self.logger.debug('Split data into train/val/test')
            train_set, valid_set, test_set = split_data(sorted_user_interactions)
            # 5. store data to cache for next use
            self.logger.debug('Save data to cache')
            pickle.dump(train_set, open(self.train_interactions_path, 'wb'))
            pickle.dump(valid_set, open(self.valid_interactions_path, 'wb'))
            pickle.dump(test_set, open(self.test_interactions_path, 'wb'))
        else:
            train_set = pickle.load(open(self.train_interactions_path, 'rb'))
            valid_set = pickle.load(open(self.valid_interactions_path, 'rb'))
            test_set = pickle.load(open(self.test_interactions_path, 'rb'))
            entities = np.load(self.entities_path, allow_pickle=True)
            user_ids = entities['user_ids']
            item_ids = entities['item_ids']
        # train interaction indexes within each user
        if not os.path.exists(self.train_interaction_indexes_path):
            self.logger.debug('Extract training interaction indexes')
            train_interaction_indexes = []
            for uid, interactions in train_set.items():
                last_idx = len(interactions) - 1
                train_interaction_indexes.append((uid, last_idx))
                if self.samples_step > 0:
                    offsets = list(range(last_idx, self.seqlen - 1, -self.samples_step))
                    for offset in offsets:
                        train_interaction_indexes.append((uid, offset))
            pickle.dump(train_interaction_indexes,
                        open(self.train_interaction_indexes_path, 'wb'))
        else:
            train_interaction_indexes = pickle.load(open(
                self.train_interaction_indexes_path, 'rb'))

        if not os.path.exists(self.train_item_pop_path):
            item_pops = self._get_item_popularities(train_set,
                                                    n_items=len(item_ids),
                                                    beta=self.neg_beta)
            np.savez(self.train_item_pop_path, item_pops=item_pops)
        else:
            item_pops = np.load(self.train_item_pop_path,
                                allow_pickle=True)['item_pops']

        self.logger.info(f'Number of users: {len(user_ids)}')
        self.logger.info(f'Number of items: {len(item_ids)}')
        n_interactions = np.sum([len(interactions) + 2
                                 for _, interactions in train_set.items()])
        self.logger.info(f'Number of interactions: {n_interactions}')
        self.logger.info(
            f'Density: {n_interactions / (len(user_ids) * len(item_ids)):3.5f}')

        # data for collaborative methods
        if not os.path.exists(self.train_aggregated_user_key_interaction_path) or \
                not os.path.exists(self.train_aggregated_item_key_interaction_path) or \
                not os.path.exists(self.train_aggregated_firsttime_path):
            self.logger.info(f'Get aggregated interactions to '
                             f'{self.train_aggregated_user_key_interaction_path}')
            train_interactions_agg_user_key_dict, \
            train_interactions_agg_item_key_dict, \
            train_firsttime_agg_dict = self._aggregated_interactions_dicts(
                train_set, train_interaction_indexes)
            pickle.dump(train_interactions_agg_user_key_dict,
                        open(self.train_aggregated_user_key_interaction_path, 'wb'))
            pickle.dump(train_interactions_agg_item_key_dict,
                        open(self.train_aggregated_item_key_interaction_path, 'wb'))
            pickle.dump(train_firsttime_agg_dict,
                        open(self.train_aggregated_firsttime_path, 'wb'))
        else:
            self.logger.info(f'Load aggregated interactions from '
                             f'{self.train_aggregated_user_key_interaction_path} and '
                             f'{self.train_aggregated_item_key_interaction_path}')
            train_interactions_agg_user_key_dict = pickle.load(open(
                self.train_aggregated_user_key_interaction_path, 'rb'))
            train_interactions_agg_item_key_dict = pickle.load(open(
                self.train_aggregated_item_key_interaction_path, 'rb'))
            train_firsttime_agg_dict = pickle.load(open(
                self.train_aggregated_firsttime_path, 'rb'))
        n_selected_interactions = 0
        for uid, iid_dict in train_interactions_agg_user_key_dict.items():
            n_selected_interactions += len(iid_dict)
        min_ts_dict = {uid: min(ts_dict.values())
                       for uid, ts_dict in train_firsttime_agg_dict.items()}
        min_ts = min(min_ts_dict.values())
        self.logger.info(f'Min timestamps in the corpus: {min_ts}')
        self.logger.info(f'Number of distinct interactions: {n_selected_interactions}')
        self.data = {
            'train_set': train_set,
            'train_interaction_indexes': train_interaction_indexes,
            'train_interactions_agg_user_key_dict': train_interactions_agg_user_key_dict,
            'train_interactions_agg_item_key_dict': train_interactions_agg_item_key_dict,
            'train_firsttime_agg_dict': train_firsttime_agg_dict,
            'train_item_popularities': item_pops,
            'min_ts': min_ts,
            'valid_set': valid_set,
            'test_set': test_set,
            'user_ids': user_ids,
            'item_ids': item_ids,
            'n_users': len(user_ids),
            'n_items': len(item_ids),
            'num_test_negatives': self.num_test_negatives,
            'num_valid_users': self.num_valid_users
        }

    def _fetch_original_kcore_interactions(self):
        dataset_name = self.dataset_params['name']
        if dataset_name == 'deezer' or dataset_name == 'lfm1b':
            org_kcore_data_path = os.path.join(
                self.dataset_params['path'],
                f'frac{self.dataset_params["fraction"]}_'
                f'interactions_{self.item_type}_'
                f'{self.u_ncore}core-users_{self.i_ncore}core-items_'
                f'loop-{self.dataset_params["loopback_interval_in_hours"]}h_'
                f'recent-{self.dataset_params["ndays_max"]}days_'
                f'histmin-{self.dataset_params["ndays_min"]}days.csv')
        else:
            org_kcore_data_path = os.path.join(
                self.dataset_params['path'],
                f'{self.dataset_params["name"]}_interactions_'
                f'{self.u_ncore}core-users_{self.i_ncore}core-items.csv')
        if not os.path.exists(org_kcore_data_path):
            # 1. read original data
            data = self._fetch_interactions()
            # 2. select only interested columns
            data = data[['org_user', f'org_{self.item_type}', 'timestamp']]
            # 3. kcore preprocessing
            self.logger.debug(f'Get {self.u_ncore}-core user, '
                              f'{self.i_ncore}-core item data')
            data = self._kcore(data, self.u_ncore, self.i_ncore)
            data.to_csv(org_kcore_data_path, sep=',',
                        header=True, index=False)
        else:
            self.logger.debug(f'Read original k-core data from {org_kcore_data_path}')
            data = pd.read_csv(org_kcore_data_path)
        return data

    def _fetch_interactions(self):
        """
        Fetch interactions from file
        """
        dataset_name = self.dataset_params['name']
        if dataset_name == 'lfm1b':
            user_interactions_path = os.path.join(
                self.dataset_params['path'],
                f'frac{self.dataset_params["fraction"]}_of_'
                f'filtered_{self.dataset_params["loopback_interval_in_hours"]}h_'
                f'loopback_interactions_'
                f'recent{self.dataset_params["ndays_max"]}days_'
                f'histmin{self.dataset_params["ndays_min"]}days.csv')
        else:
            user_interactions_path = os.path.join(
                self.dataset_params['path'],
                f'frac{self.dataset_params["fraction"]}_of_filtered_'
                f'{self.dataset_params["loopback_interval_in_hours"]}h_'
                f'loopback_interactions_recent{self.dataset_params["ndays_max"]}days_'
                f'histmin{self.dataset_params["ndays_min"]}days.csv')
        self.logger.info(f'Fetch information from {user_interactions_path}')
        data = pd.read_csv(user_interactions_path,
                           names=self.dataset_params['col_names'])
        return data

    def _kcore(self, data, u_ncore, i_ncore):
        """
        Preprocessing data to get k-core dataset.
        Each user has at least u_ncore items in his preference
        Each item is interacted by at least i_ncore users
        :param data:
        :param u_ncore: min number of interactions for each user
        :param i_ncore: min number of interactions for each item

        :return:
        """
        if u_ncore <= 1 and i_ncore <= 1:
            return data

        def filter_user(df):
            """ Filter out users less than u_ncore interactions """
            tmp = df.groupby(['org_user'], as_index=False)[f'org_{self.item_type}'].count()
            tmp.rename(columns={f'org_{self.item_type}': 'cnt_item'},
                       inplace=True)
            df = df.merge(tmp, on=['org_user'])
            df = df[df['cnt_item'] >= u_ncore].reset_index(drop=True).copy()
            df.drop(['cnt_item'], axis=1, inplace=True)
            return df

        def filter_item(df):
            """ Filter out items less than u_ncore interactions """
            tmp = df.groupby([f'org_{self.item_type}'], as_index=False)['org_user'].count()
            tmp.rename(columns={'org_user': 'cnt_user'},
                       inplace=True)
            df = df.merge(tmp, on=[f'org_{self.item_type}'])
            df = df[df['cnt_user'] >= i_ncore].reset_index(drop=True).copy()
            df.drop(['cnt_user'], axis=1, inplace=True)
            return df

        # because of repeat consumption, just count 1 for each user-item pair
        unique_data = data[['org_user', f'org_{self.item_type}']].drop_duplicates()
        while 1:
            unique_data = filter_user(unique_data)
            unique_data = filter_item(unique_data)
            chk_u = unique_data.groupby('org_user')[f'org_{self.item_type}'].count()
            chk_i = unique_data.groupby(f'org_{self.item_type}')['org_user'].count()
            if len(chk_i[chk_i < i_ncore]) <= 0 and len(chk_u[chk_u < u_ncore]) <= 0:
                break

        unique_data = unique_data.dropna()
        data = pd.merge(data, unique_data, on=['org_user', f'org_{self.item_type}'])
        return data

    def _map_internal_ids(self, data):
        data.rename(columns={f'org_{self.item_type}': 'org_item'},
                    inplace=True)
        user_ids = data['org_user'].drop_duplicates().to_numpy()
        item_ids = data['org_item'].drop_duplicates().to_numpy()
        # map to internal ids
        user_id_map = {uid: idx + 1 for idx, uid in enumerate(user_ids)}
        item_id_map = {iid: idx + 1 for idx, iid in enumerate(item_ids)}
        data.loc[:, 'user'] = data.org_user.apply(lambda x: user_id_map[x])
        data.loc[:, 'item'] = data.org_item.apply(lambda x: item_id_map[x])
        data = data.drop(columns=['org_user', f'org_item'])
        return data, user_ids, item_ids

    def _aggregated_interactions_dicts(self, dataset, indexes):
        i_freq_dict = defaultdict(float_dd)
        u_freq_dict = defaultdict(float_dd)
        first_time_dict = defaultdict(dict)
        for uid, nxt_idx in indexes:
            idx = self.seqlen - 1
            for iid, ts in reversed(dataset[uid][:nxt_idx]):
                i_freq_dict[uid][iid] += 1.
                u_freq_dict[iid][uid] += 1.
                if iid not in first_time_dict[uid] or ts < first_time_dict[uid][iid]:
                    first_time_dict[uid][iid] = ts
                idx -= 1
                if idx == -1:
                    break
            iid, ts = dataset[uid][nxt_idx]
            i_freq_dict[uid][iid] += 1.
            u_freq_dict[iid][uid] += 1.
            if iid not in first_time_dict[uid] or ts < first_time_dict[uid][iid]:
                first_time_dict[uid][iid] = ts
        return i_freq_dict, u_freq_dict, first_time_dict

    def _get_item_popularities(self, dataset, n_items, beta=1.0):
        self.logger.debug('Get item popularity')
        item_popularities = np.zeros(n_items, dtype=np.float32)
        for uid, interactions in dataset.items():
            for iid, _ in interactions:
                item_popularities[iid - 1] += 1.0
        if beta != 1.0:
            item_popularities = item_popularities ** beta
        total_counts = np.sum(item_popularities)
        item_popularities = item_popularities / total_counts
        return item_popularities
