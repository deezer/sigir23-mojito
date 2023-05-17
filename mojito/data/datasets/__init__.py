from mojito.data.datasets.dataset import Dataset
from mojito.data.datasets.tempo.moji import MojitoDataset


_SUPPORTED_DATASETS = {
    'pos': Dataset,
    'tempo_mojito': MojitoDataset
}


def dataset_factory(params):
    """
    Factory that generate dataset
    :param params:
    :return:
    """
    dataloader_type = params['dataset'].get('dataloader', 'pos')
    try:
        return _SUPPORTED_DATASETS[dataloader_type](params).data
    except KeyError:
        raise KeyError(f'Not support {dataloader_type} dataset')
