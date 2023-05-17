import os
from mojito import MojitoError


def process_params(params):
    dataset_params = params['dataset']
    training_params = params['training']
    model_params = training_params['model']

    dataset_spec = f'{dataset_params["name"]}_' \
                   f'{dataset_params["u_ncore"]}ucore_' \
                   f'{dataset_params["i_ncore"]}icore'
    training_spec = f'lr{training_params["learning_rate"]}_' \
                    f'batch{training_params["batch_size"]}_' \
                    f'dim{training_params["embedding_dim"]}'
    model_type = model_params['type']
    model_name = model_params['name']

    if model_name == 'mojito':
        model_spec = f'{model_type}_{model_name}_' \
                     f'{training_spec}_' \
                     f'seqlen{model_params["params"]["seqlen"]}_' \
                     f'l2emb{model_params["params"]["l2_emb"]}_' \
                     f'nblocks{model_params["params"]["num_blocks"]}_' \
                     f'nheads{model_params["params"]["num_heads"]}_' \
                     f'dropout{model_params["params"]["dropout_rate"]}_' \
                     f'tempo-dim{training_params["tempo_embedding_dim"]}-' \
                     f'linspace{training_params["tempo_linspace"]}'
        if 'lambda_trans_seq' in model_params["params"]:
            model_spec = f'{model_spec}_' \
                         f'lbdatrans{model_params["params"]["lambda_trans_seq"]}'
        if 'causality' in model_params['params'] and \
                model_params['params']['causality'] is False:
            model_spec = f'{model_spec}_noncausal'
        model_spec = f'{model_spec}_mercer2_' \
                     f'linspace{training_params["tempo_linspace"]}'
        if 'sse' in model_params['params'] and \
                model_params['params']['sse']['activate'] is True:
            model_spec = f'{model_spec}_' \
                         f'sse-{model_params["params"]["sse"]["type"]}_' \
                         f'thresitem{model_params["params"]["sse"]["threshold_item"]}'
            if 'threshold_user' in model_params["params"]["sse"]:
                model_spec = f'{model_spec}_' \
                             f'thresuser{model_params["params"]["sse"]["threshold_user"]}'
            if model_params["params"]["sse"]["type"] != 'uniform':
                model_spec = f'{model_spec}_' \
                             f'thresfavs{model_params["params"]["sse"]["threshold_favs"]}'
        if 'fism' in model_params['params']:
            fism_params = model_params['params']['fism']
            fism_spec = f'fism_{fism_params["sampling"]}_' \
                        f'{fism_params["n_items"]}items_' \
                        f'beta-{fism_params["beta"]}'
            model_spec = f'{model_spec}_{fism_spec}'
        if 'residual' in model_params['params']:
            model_spec = f'{model_spec}_residual-{model_params["params"]["residual"]}'
        if 'use_year' in model_params['params'] and \
                model_params['params']['use_year'] is False:
            model_spec = f'{model_spec}_noyear'
        if 'lambda_glob' in model_params['params']:
            model_spec = f'{model_spec}_' \
                         f'glob{model_params["params"]["lambda_glob"]}'
        if 'lambda_user' in model_params['params']:
            model_spec = f'{model_spec}_' \
                         f'l2u{model_params["params"]["lambda_user"]}'
        if 'lambda_item' in model_params['params']:
            model_spec = f'{model_spec}_' \
                         f'l2i{model_params["params"]["lambda_item"]}'
    else:
        raise MojitoError(f'Unknown model name {model_name}')

    training_params['model_dir'] = os.path.join(
        training_params['model_dir'],
        dataset_spec,
        f'samples_step{dataset_params["samples_step"]}',
        f'nepoch{training_params["num_epochs"]}',
        model_spec)
    return training_params, model_params
