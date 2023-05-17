from mojito.data.loaders.loader import DataLoader
from mojito.data.samplers import one_test_sample


class TestDataLoader(DataLoader):
    def __init__(self, dataset, n_users, n_items,
                 batch_size, seqlen, need_neg_dataset=False,
                 random_seed=2022, **kwargs):
        super(TestDataLoader, self).__init__(
            dataset, n_users, n_items, batch_size, seqlen,
            need_neg_dataset=need_neg_dataset,
            random_seed=random_seed, **kwargs)
        self.user_ids = list(dataset.keys())
        if kwargs['num_scored_users'] > 0:
            self.rng.shuffle(self.user_ids)
            self.user_ids = self.user_ids[:kwargs['num_scored_users']]
            self.dataset = {uid: self.dataset[uid] for uid in self.user_ids}
        self.n_batches = int(len(self.dataset) / self.batch_size)
        if self.n_batches * self.batch_size < len(self.dataset):
            self.n_batches += 1

    def _batch_sampling(self, batch_index):
        batch_user_ids = self.user_ids[batch_index * self.batch_size:
                                       (batch_index + 1) * self.batch_size]
        return self._batch_sampling_seq(batch_user_ids)

    def _batch_sampling_seq(self, batch_user_ids):
        """
        Batch sampling
        :param batch_user_ids:
        :return:
        """
        output = []
        for uid in batch_user_ids:
            one_sample = one_test_sample(self.model_name,
                                         uid, self.dataset, self.seqlen,
                                         self.n_items, **self.kwargs)
            output.append(one_sample)
        return list(zip(*output))
