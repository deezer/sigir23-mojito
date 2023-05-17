from mojito.data.loaders.loader import DataLoader
from mojito.data.samplers import one_train_sample


class TrainDataLoader(DataLoader):
    def __init__(self, dataset, n_users, n_items,
                 batch_size, seqlen, need_neg_dataset=False,
                 random_seed=2022, **kwargs):
        super(TrainDataLoader, self).__init__(
            dataset, n_users, n_items, batch_size, seqlen,
            need_neg_dataset=need_neg_dataset,
            random_seed=random_seed, **kwargs)
        self.interaction_indexes = \
            kwargs['train_interaction_indexes']
        self.rng.shuffle(self.interaction_indexes)
        self.n_batches = int(len(self.interaction_indexes) / batch_size)
        if self.n_batches * self.batch_size < len(self.interaction_indexes):
            self.n_batches += 1

    def _batch_sampling(self, batch_index):
        batch_interaction_indexes = self.interaction_indexes[
                                    batch_index * self.batch_size:
                                    (batch_index + 1) * self.batch_size]
        return self._batch_sampling_seq(batch_interaction_indexes)

    def _batch_sampling_seq(self, batch_interaction_indexes):
        """
        Batch sampling
        :param batch_interaction_indexes:
        :return:
        """
        output = []
        for uid, idx in batch_interaction_indexes:
            one_sample = one_train_sample(self.model_name,
                                          uid, idx, self.dataset, self.seqlen,
                                          self.n_items,
                                          **self.kwargs)
            output.append(one_sample)
        return list(zip(*output))
