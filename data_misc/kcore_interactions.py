import os
import json
import logging
import pandas as pd

_FORMAT = '%(asctime)s:%(levelname)s:%(name)s:%(message)s'


class _LoggerHolder(object):
    """
    Logger singleton instance holder.
    """
    INSTANCE = None


def get_logger():
    """
    Returns library scoped logger.
    :returns: Library logger.
    """
    if _LoggerHolder.INSTANCE is None:
        formatter = logging.Formatter(_FORMAT)
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger = logging.getLogger('repeatflow')
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        _LoggerHolder.INSTANCE = logger
    return _LoggerHolder.INSTANCE


def load_configuration(descriptor):
    """
    Load configuration from the given descriptor.
    Args:
        descriptor:
    Returns:
    """
    if not os.path.exists(descriptor):
        raise IOError(f'Configuration file {descriptor} '
                      f'not found')
    with open(descriptor, 'r') as stream:
        return json.load(stream)


def kcore(data, u_ncore, i_ncore):
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
        tmp = df.groupby(['org_user'], as_index=False)['org_item'].count()
        tmp.rename(columns={'org_item': 'cnt_item'},
                   inplace=True)
        df = df.merge(tmp, on=['org_user'])
        df = df[df['cnt_item'] >= u_ncore].reset_index(drop=True).copy()
        df.drop(['cnt_item'], axis=1, inplace=True)
        return df

    def filter_item(df):
        """ Filter out items less than u_ncore interactions """
        tmp = df.groupby(['org_item'], as_index=False)['org_user'].count()
        tmp.rename(columns={'org_user': 'cnt_user'},
                   inplace=True)
        df = df.merge(tmp, on=['org_item'])
        df = df[df['cnt_user'] >= i_ncore].reset_index(drop=True).copy()
        df.drop(['cnt_user'], axis=1, inplace=True)
        return df

    # because of repeat consumption, just count 1 for each user-item pair
    unique_data = data[['org_user', 'org_item']].drop_duplicates()
    while 1:
        unique_data = filter_user(unique_data)
        unique_data = filter_item(unique_data)
        chk_u = unique_data.groupby('org_user')['org_item'].count()
        chk_i = unique_data.groupby('org_item')['org_user'].count()
        if len(chk_i[chk_i < i_ncore]) <= 0 and len(chk_u[chk_u < u_ncore]) <= 0:
            break

    unique_data = unique_data.dropna()
    data = pd.merge(data, unique_data, on=['org_user', 'org_item'])
    return data


# change another configuration file to preprocess the corresponding dataset
# for example amazon book: configs/amzb.json
params = load_configuration(f'configs/ml1m.json')
dataset_params = params['dataset']
u_ncore = dataset_params.get('u_ncore', 1)
i_ncore = dataset_params.get('i_ncore', 1)

logger = get_logger()

data_path = os.path.join(dataset_params['path'],
                         f'{dataset_params["interactions"]}.{dataset_params["file_format"]}')
output_path = os.path.join(
            dataset_params['path'],
            f'{dataset_params["name"]}_interactions_'
            f'{u_ncore}core-users_{i_ncore}core-items.csv')

if not os.path.exists(output_path):
    logger.info(f'Read data from csv {data_path}')
    data = pd.read_csv(data_path, sep=dataset_params['sep'],
                       names=dataset_params['col_names'])

    logger.info(f'k-core extraction with u_ncore={u_ncore} & i_ncore={i_ncore}')
    data = kcore(data, u_ncore=u_ncore, i_ncore=i_ncore)
    logger.info(f'Write to {output_path}')
    data.to_csv(output_path, sep=',', header=True, index=False)
    logger.info('Finish')
else:
    logger.info(f'Read data from csv {output_path}')
    data = pd.read_csv(output_path)

logger.info(f'Number of users: {len(data["org_user"].unique())}')
logger.info(f'Number of items: {len(data["org_item"].unique())}')
logger.info(f'Number of interactions: {len(data)}')
