import random
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
    # their corresponding timestamps
    in_ts_seq = np.zeros([seqlen], dtype=np.int32)
    # sequence of target timestamps
    nxt_ts_seq = np.zeros([seqlen], dtype=np.int32)
    # sequence of target items
    pos = np.zeros([seqlen], dtype=np.int32)
    # sequence of random negative items
    neg = np.zeros([seqlen], dtype=np.int32)
    nxt = dataset[uid][nxt_idx][0]
    nxt_time = dataset[uid][nxt_idx][1]

    activate_sse = kwargs['activate_sse']
    sse_type = kwargs['sse_type']
    threshold_item = kwargs['threshold_item'] \
        if 'threshold_item' in kwargs else 1.
    threshold_favs = kwargs['threshold_favs'] \
        if 'threshold_favs' in kwargs else 1.
    threshold_user = kwargs['threshold_user'] \
        if 'threshold_user' in kwargs else 1.
    idx = seqlen - 1
    # list of historic items
    favs = set(map(lambda x: x[0], dataset[uid]))
    favs_list = list(favs)
    for interaction in reversed(dataset[uid][:nxt_idx]):
        iid, ts = interaction
        if activate_sse is True:
            if sse_type == 'uniform':
                if random.random() > threshold_item:
                    iid = np.random.randint(1, n_items + 1)
                    nxt = np.random.randint(1, n_items + 1)
            else:
                p_favs = random.random()
                if p_favs > threshold_favs:
                    iid = np.random.choice(favs_list)
                    nxt = np.random.choice(favs_list)
                elif random.random() > threshold_item:
                    iid = np.random.randint(1, n_items + 1)
                    nxt = np.random.randint(1, n_items + 1)
        seq[idx] = iid
        in_ts_seq[idx] = ts
        pos[idx] = nxt
        nxt_ts_seq[idx] = nxt_time
        if nxt != 0:
            neg[idx] = random_neg(1, n_items + 1, favs)
        nxt = iid
        nxt_time = ts
        idx -= 1
        if idx == -1:
            break
    if random.random() > threshold_user:
        uid = np.random.randint(1, kwargs['n_users'] + 1)
    out = uid, seq, pos, neg
    if 'time_dict' in kwargs and kwargs['time_dict'] is not None:
        seq_year = np.zeros([seqlen], dtype=np.int32)
        seq_month = np.zeros([seqlen], dtype=np.int32)
        seq_day = np.zeros([seqlen], dtype=np.int32)
        seq_dayofweek = np.zeros([seqlen], dtype=np.int32)
        seq_dayofyear = np.zeros([seqlen], dtype=np.int32)
        seq_week = np.zeros([seqlen], dtype=np.int32)
        seq_hour = np.zeros([seqlen], dtype=np.int32)
        nxt_year = np.zeros([seqlen], dtype=np.int32)
        nxt_month = np.zeros([seqlen], dtype=np.int32)
        nxt_day = np.zeros([seqlen], dtype=np.int32)
        nxt_dayofweek = np.zeros([seqlen], dtype=np.int32)
        nxt_dayofyear = np.zeros([seqlen], dtype=np.int32)
        nxt_week = np.zeros([seqlen], dtype=np.int32)
        nxt_hour = np.zeros([seqlen], dtype=np.int32)
        for i, ts in enumerate(in_ts_seq):
            if ts > 0:
                seq_year[i], seq_month[i], seq_day[i], seq_dayofweek[i], \
                    seq_dayofyear[i], seq_week[i], seq_hour[i] = kwargs['time_dict'][ts]
        for i, ts in enumerate(nxt_ts_seq):
            if ts > 0:
                nxt_year[i], nxt_month[i], nxt_day[i], nxt_dayofweek[i], \
                    nxt_dayofyear[i], nxt_week[i], nxt_hour[i] = kwargs['time_dict'][ts]
        out = out + (seq_year, seq_month, seq_day,
                     seq_dayofweek, seq_dayofyear, seq_week, seq_hour,
                     nxt_year, nxt_month, nxt_day,
                     nxt_dayofweek, nxt_dayofyear, nxt_week, nxt_hour)
    if 'item_fism' in kwargs:
        item_fism_seq = kwargs['item_fism'][uid]
        if len(item_fism_seq) < kwargs['n_fism_items']:
            zeros = [0] * (kwargs['n_fism_items'] - len(item_fism_seq))
            item_fism_seq = item_fism_seq + tuple(zeros)
        out = out + (item_fism_seq,)
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
    ts_seq = np.zeros([seqlen], dtype=np.int32)

    idx = seqlen - 1
    for interaction in reversed(train_set[uid]):
        seq[idx] = interaction[0]
        ts_seq[idx] = interaction[1]
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
        zeros = np.array(list(rated)) - 1
        p = kwargs['train_item_popularities'].copy()
        p[zeros] = 0.0
        p = p / p.sum()
        neg_item_ids = np.random.choice(range(1, n_items + 1),
                                        size=num_negatives,
                                        p=p,
                                        replace=False)
        test_item_ids = test_item_ids + neg_item_ids.tolist()
    out = uid, seq, test_item_ids
    if 'time_dict' in kwargs and kwargs['time_dict'] is not None:
        test_ts = dataset[uid][0][1]
        seq_year = np.zeros([seqlen], dtype=np.int32)
        seq_month = np.zeros([seqlen], dtype=np.int32)
        seq_day = np.zeros([seqlen], dtype=np.int32)
        seq_dayofweek = np.zeros([seqlen], dtype=np.int32)
        seq_dayofyear = np.zeros([seqlen], dtype=np.int32)
        seq_week = np.zeros([seqlen], dtype=np.int32)
        seq_hour = np.zeros([seqlen], dtype=np.int32)
        for i, ts in enumerate(ts_seq):
            if ts > 0:
                seq_year[i], seq_month[i], seq_day[i], seq_dayofweek[i], \
                    seq_dayofyear[i], seq_week[i], seq_hour[i] = kwargs['time_dict'][ts]
        test_year, test_month, test_day, test_dayofweek, \
            test_dayofyear, test_week, test_hour = kwargs['time_dict'][test_ts]
        out = out + (seq_year, seq_month, seq_day,
                     seq_dayofweek, seq_dayofyear, seq_week, seq_hour,
                     test_year, test_month, test_day,
                     test_dayofweek, test_dayofyear, test_week, test_hour)
    if 'item_fism' in kwargs:
        item_fism_seq = kwargs['item_fism'][uid]
        if len(item_fism_seq) < kwargs['n_fism_items']:
            zeros = [0] * (kwargs['n_fism_items'] - len(item_fism_seq))
            item_fism_seq = item_fism_seq + tuple(zeros)
        out = out + (item_fism_seq,)
    return out
