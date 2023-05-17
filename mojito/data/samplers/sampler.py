import numpy as np

from mojito.utils import random_neg


def train_sample(uid, nxt_idx, dataset, seqlen, n_items,
                 **kwargs):
    """
    Sampling train data for a given user
    :param uid: user id
    :param nxt_idx: next interaction index
    :param dataset: dataset
    :param seqlen: sequence length
    :param n_items: number of items
    :param kwargs: additional parameters
    :return:
    """
    # sequence of previous items
    seq = np.zeros([seqlen], dtype=np.int32)
    # sequence of target items
    pos = np.zeros([seqlen], dtype=np.int32)
    # sequence of random negative items
    neg = np.zeros([seqlen], dtype=np.int32)
    nxt = dataset[uid][nxt_idx][0]

    idx = seqlen - 1
    # list of historic items
    favs = set(map(lambda x: x[0], dataset[uid]))
    for interaction in reversed(dataset[uid][:nxt_idx]):
        seq[idx] = interaction[0]
        pos[idx] = nxt
        if nxt != 0:
            neg[idx] = random_neg(1, n_items + 1, favs)
        nxt = interaction[0]
        idx -= 1
        if idx == -1:
            break
    out = uid, seq, pos, neg
    # uid, seq, pos, neg
    return out


def test_sample(uid, dataset, seqlen, n_items, **kwargs):
    """
    Sampling test data for a given user
    :param uid:
    :param dataset:
    :param seqlen:
    :param n_items:
    :param kwargs:
    :return:
    """
    train_set = kwargs['train_set']
    num_negatives = kwargs['num_test_negatives']
    seq = np.zeros([seqlen], dtype=np.int32)

    idx = seqlen - 1
    for interaction in reversed(train_set[uid]):
        seq[idx] = interaction[0]
        idx -= 1
        if idx == -1:
            break
    rated = set([x[0] for x in train_set[uid]])
    rated.add(dataset[uid][0][0])
    rated.add(0)
    # list of test items, beginning with positive
    # and then negative items
    test_item_ids = [dataset[uid][0][0]]
    neg_sampling = kwargs['neg_sampling']
    if neg_sampling == 'uniform':
        for _ in range(num_negatives):
            t = np.random.randint(1, n_items + 1)
            while t in rated:
                t = np.random.randint(1, n_items + 1)
            test_item_ids.append(t)
    else:
        neg_item_ids = np.random.choice(range(1, n_items + 1),
                                        size=num_negatives,
                                        p=kwargs['train_item_popularities'])
        for j, neg_id in enumerate(neg_item_ids):
            while neg_id in rated:
                neg_item_ids[j] = neg_id = np.random.choice(
                    range(1, n_items + 1), p=kwargs['train_item_popularities'])
        test_item_ids = test_item_ids + neg_item_ids.tolist()
    out = uid, seq, test_item_ids
    return out
