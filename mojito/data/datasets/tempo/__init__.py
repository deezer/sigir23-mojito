from tqdm import tqdm

from mojito.data.datasets.dataset import Dataset


class TempoDataset(Dataset):
    def __init__(self, params):
        super(TempoDataset, self).__init__(params)
        self.model_params = params['training']['model']['params']
        self.dataset_params = params['dataset']
        self.cache_params = params['cache']
        # fetch additional temporal data
        self._fetch_tempo_data()

    def _fetch_tempo_data(self):
        raise NotImplementedError('_fetch_tempo_data method should be '
                                  'implemented in concrete model')

    def _time_set(self, dataset, indexes=None):
        ts_set = set()
        n_users = len(dataset)
        cnt = 0
        if indexes is None:
            for _, interactions in dataset.items():
                for interaction in interactions:
                    ts_set.add(interaction[1])
                cnt += 1
                if cnt % 5000 == 0 or cnt == n_users:
                    self.logger.debug(f'----> {cnt} / {n_users} users')
        else:
            for uid, nxt_idx in tqdm(indexes, desc='Generate time set'):
                idx = self.seqlen - 1
                for _, ts in reversed(dataset[uid][:nxt_idx]):
                    if ts > 0:
                        ts_set.add(ts)
                    idx -= 1
                    if idx == -1:
                        break
                _, ts = dataset[uid][nxt_idx]
                ts_set.add(ts)
        return ts_set
