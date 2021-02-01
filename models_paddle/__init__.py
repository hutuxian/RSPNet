from typing import *
import logging

logger = logging.getLogger(__name__)


def get_model_class(arch):
    """
    return model by arch name
    """
    logger.info('Using global get_model_class(%s)' % (arch))

    if arch == 'resnet18':
        from .resnet import resnet18
        model_class = resnet18
    elif arch == 'resnet34':
        from .resnet import resnet34
        model_class = resnet34
    elif arch == 'resnet50':
        from .resnet import resnet50
        model_class = resnet50
    else:
        raise ValueError('Unknown model architecture "{%s}"' % (arch))

    return model_class
