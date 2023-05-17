from mojito import MojitoError
from mojito.data.loaders.train_loader import TrainDataLoader
from mojito.data.loaders.test_loader import TestDataLoader


_SUPPORTED_DATALOADERS = {
    'train': TrainDataLoader,
    'valid': TestDataLoader,
    'test': TestDataLoader
}

_SUPPORTED_KWARGS = ['train_interactions_agg_user_key_dict',
                     'train_interactions_agg_item_key_dict',
                     'min_ts',
                     'num_test_negatives', 'num_valid_users',
                     'time_relation_matrix', 'train_item_popularities']


def dataloader_factory(data, batch_size, seqlen, mode='train',
                       random_seed=2022, num_scored_users=-1,
                       cache_path='', spec='', epoch=-1,
                       model_name=None, timespan=256,
                       fism_sampling='pop', n_fism_items=-1,
                       fism_type='item',
                       fism_beta=1.0,
                       frac=0.05,
                       ctx_size=6,
                       activate_sse=False,
                       sse_type='uniform',
                       threshold_item=1.0,
                       threshold_favs=1.0,
                       threshold_user=1.0,
                       neg_sampling='uniform',
                       train_interaction_indexes=None):
    """
    Create a data loader for train/valid/test
    :param data:
    :param batch_size:
    :param seqlen:
    :param mode:
    :param random_seed:
    :param num_scored_users:
    :param cache_path:
    :param spec:
    :param epoch:
    :param model_name:
    :param timespan:
    :param fism_sampling:
    :param n_fism_items:
    :param fism_type:
    :param fism_beta:
    :param frac:
    :param ctx_size:
    :param activate_sse:
    :param sse_type:
    :param threshold_item:
    :param threshold_favs:
    :param threshold_user:
    :param neg_sampling:
    :param train_interaction_indexes:
    :return:
    """
    kwargs = {key: data[key] if key in data else None
              for key in _SUPPORTED_KWARGS}
    kwargs['frac'] = frac
    kwargs['num_scored_users'] = num_scored_users
    kwargs['mode'] = mode
    kwargs['epoch'] = epoch
    kwargs['cache_path'] = cache_path
    kwargs['spec'] = spec
    kwargs['model_name'] = model_name
    kwargs['timespan'] = timespan
    kwargs['neg_sampling'] = neg_sampling
    if n_fism_items > 0:
        kwargs['fism_type'] = fism_type
        kwargs['n_fism_items'] = n_fism_items
        kwargs['fism_sampling'] = fism_sampling
        kwargs['fism_beta'] = fism_beta
    if mode == 'train':
        kwargs['train_interaction_indexes'] = train_interaction_indexes
    if 'time_dict' in data:
        kwargs['time_dict'] = data['time_dict']
    kwargs['ctx_size'] = ctx_size
    kwargs['activate_sse'] = activate_sse
    kwargs['sse_type'] = sse_type
    kwargs['threshold_item'] = threshold_item
    kwargs['threshold_favs'] = threshold_favs
    kwargs['threshold_user'] = threshold_user
    if mode == 'valid':
        kwargs['train_set'] = data['train_set']
        kwargs['train_firsttime_dict'] = data['train_firsttime_agg_dict']
    elif mode == 'test':
        kwargs['train_set'] = {uid: iids + data['valid_set'][uid]
                               for uid, iids in data['train_set'].items()}
        train_firsttime_agg_dict = data['train_firsttime_agg_dict']
        for uid, interactions in data['valid_set'].items():
            iid, ts = interactions[0]
            if iid not in train_firsttime_agg_dict[uid]:
                train_firsttime_agg_dict[uid][iid] = ts
            kwargs['train_interactions_agg_user_key_dict'][uid][iid] += 1.
            kwargs['train_interactions_agg_item_key_dict'][iid][uid] += 1.
        kwargs['train_firsttime_dict'] = train_firsttime_agg_dict
    try:
        return _SUPPORTED_DATALOADERS[mode](data[f'{mode}_set'],
                                            n_users=data['n_users'],
                                            n_items=data['n_items'],
                                            batch_size=batch_size,
                                            seqlen=seqlen,
                                            random_seed=random_seed,
                                            **kwargs)
    except KeyError as err:
        raise MojitoError(f'{err}')
