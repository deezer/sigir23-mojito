from mojito import MojitoError


def one_train_sample(model, uid, nxt_idx, dataset, seqlen, n_items,
                     **kwargs):
    if model == 'sasrec':
        from mojito.data.samplers.sampler import train_sample
    elif 'mojito' in model:
        from mojito.data.samplers.mojito_sampler import train_sample
    else:
        raise MojitoError(f'Not support train sampler for {model} model')
    return train_sample(uid, nxt_idx, dataset, seqlen, n_items,
                        **kwargs)


def one_test_sample(model, uid, dataset, seqlen, n_items, **kwargs):
    if model == 'sasrec':
        from mojito.data.samplers.sampler import test_sample
    elif 'mojito' in model:
        from mojito.data.samplers.mojito_sampler import test_sample
    else:
        raise MojitoError(f'Not support test sampler for {model} model')
    return test_sample(uid, dataset, seqlen, n_items, **kwargs)
