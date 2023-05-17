import numpy as np
import pandas as pd
from tqdm import tqdm


def ts_to_high_level_tempo(time_set):
    df = pd.DataFrame(list(time_set), columns=['ts'])
    df['dt'] = pd.to_datetime(df['ts'], unit='s')
    df = df.sort_values(by=['ts'])
    df['year'], df['month'], df['day'], df['dayofweek'], df['dayofyear'], df['week'], df['hour'] = zip(
        *df['dt'].map(lambda x: [x.year, x.month, x.day, x.dayofweek, x.dayofyear, x.week, x.hour]))
    df['year'] -= df['year'].min()
    df['year'] += 1
    df['dayofweek'] += 1
    res = {}
    for _, row in df.iterrows():
        ts = row['ts']
        year = row['year']
        month = row['month']
        day = row['day']
        dayofweek = row['dayofweek']
        dayofyear = row['dayofyear']
        week = row['week']
        hour = row['hour']
        res[ts] = year, month, day, dayofweek, dayofyear, week, hour
    return res


def ts_to_carca_tempo(time_set):
    df = pd.DataFrame(list(time_set), columns=['ts'])
    df['dt'] = pd.to_datetime(df['ts'], unit='s')
    df = df.sort_values(by=['ts'])
    df['year'], df['month'], df['day'], df['dayofweek'], df['dayofyear'], df['week'] = zip(
        *df['dt'].map(lambda x: [x.year, x.month, x.day, x.dayofweek, x.dayofyear, x.week]))
    df['year'] -= df['year'].min()
    df['year'] /= df['year'].max()
    df['month'] /= 12
    df['day'] /= 31
    df['dayofweek'] /= 7
    df['dayofyear'] /= 365
    df['week'] /= 4
    res = {}
    for _, row in df.iterrows():
        ts = row['ts']
        year = row['year']
        month = row['month']
        day = row['day']
        dayofweek = row['dayofweek']
        dayofyear = row['dayofyear']
        week = row['week']
        res[ts] = np.array([year, month, day, dayofweek, dayofyear, week])
    return res


def normalize_time(time_list):
    time_diff = set()
    for i in range(len(time_list) - 1):
        if time_list[i + 1] - time_list[i] != 0:
            time_diff.add(time_list[i + 1] - time_list[i])
    if len(time_diff) == 0:
        time_scale = 1
    else:
        time_scale = min(time_diff)
    time_min = min(time_list)
    res = [int(round((t - time_min) / time_scale) + 1) for t in time_list]
    return res


def compute_time_matrix(time_seq, time_span):
    """
    Compute temporal relation matrix for the given time sequence
    :param time_seq: Timestamp sequence
    :param time_span: threshold
    :return:
    """
    size = time_seq.shape[0]
    time_matrix = np.zeros([size, size], dtype=np.int32)
    for i in range(size):
        for j in range(size):
            span = abs(time_seq[i] - time_seq[j])
            if time_span is not None and span > time_span:
                time_matrix[i][j] = time_span
            else:
                time_matrix[i][j] = span
    return time_matrix


def compute_relation_matrix(user_train, usernum, maxlen, time_span):
    """
    Compute temporal relation matrix for all users
    :param user_train:
    :param usernum:
    :param maxlen:
    :param time_span:
    :return:
    """
    data_train = dict()
    for user in tqdm(range(1, usernum + 1),
                     desc='Preparing temporal relation matrix'):
        time_seq = np.zeros([maxlen], dtype=np.int32)
        idx = maxlen - 1
        for i in reversed(user_train[user][:-1]):
            time_seq[idx] = i[1]
            idx -= 1
            if idx == -1:
                break
        data_train[user] = compute_time_matrix(time_seq, time_span)
    return data_train
