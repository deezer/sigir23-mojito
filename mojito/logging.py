import logging


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
        logger = logging.getLogger('mojito')
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        _LoggerHolder.INSTANCE = logger
    return _LoggerHolder.INSTANCE


def enable_verbose_logging():
    """ Enable tensorflow logging. """
    logger = get_logger()
    logger.setLevel(logging.DEBUG)
