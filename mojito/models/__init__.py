from mojito import MojitoError
from mojito.models.net import Mojito

_SUPPORTED_MODELS = {
    'mojito': Mojito
}


class ModelFactory:
    @classmethod
    def generate_model(cls, sess, params, n_users, n_items,
                       command='train'):
        """
        Factory method to generate a model
        :param sess:
        :param params:
        :param n_users:
        :param n_items:
        :param command:
        :return:
        """
        model_name = params['model']['name']
        try:
            # create a new model
            mdl = _SUPPORTED_MODELS[model_name](sess=sess,
                                                params=params,
                                                n_users=n_users,
                                                n_items=n_items)
            if command == 'train':
                # build computation graph
                mdl.build_graph(name=model_name)
            elif command == 'eval':
                mdl.restore(name=model_name)
            return mdl
        except KeyError:
            raise MojitoError(f'Currently not support model {model_name}')
