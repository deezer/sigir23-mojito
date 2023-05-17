import json
import os

from mojito import MojitoError


def load_configuration(descriptor):
    """
    Load configuration from the given descriptor.
    Args:
        descriptor:
    Returns:
    """
    if not os.path.exists(descriptor):
        raise MojitoError(f'Configuration file {descriptor} '
                          f'not found')
    with open(descriptor, 'r') as stream:
        return json.load(stream)
