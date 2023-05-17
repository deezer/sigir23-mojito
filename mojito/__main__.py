import sys
import warnings

from mojito import MojitoError
from mojito.commands import create_argument_parser
from mojito.configuration import load_configuration
from mojito.logging import get_logger, enable_verbose_logging


def main(argv):
    try:
        parser = create_argument_parser()
        arguments = parser.parse_args(argv[1:])
        if arguments.verbose:
            enable_verbose_logging()
        if arguments.command == 'train':
            from mojito.commands.train import entrypoint
        elif arguments.command == 'eval':
            from mojito.commands.eval import entrypoint
        else:
            raise MojitoError(
                f'mojito does not support command {arguments.command}')
        params = load_configuration(arguments.configuration)
        params['best_epoch'] = arguments.best_epoch
        entrypoint(params)
    except MojitoError as e:
        get_logger().error(e)


def entrypoint():
    """ Command line entrypoint. """
    warnings.filterwarnings('ignore')
    main(sys.argv)


if __name__ == '__main__':
    entrypoint()
