# -*- coding: utf-8 -*-

"""
Utilities for transforming loaded data for different tasks.
"""

# Basic import(s)
import numpy as np
from collections import namedtuple


def regularise (tf):
    """
    Function decorator to regularise a (partially) transformed batch of data.

    Arguments:
        tf: Base transform, assumed to return a numpy recarray.

    Returns:
        Regular numpy array with dtype '<f8'.
    """

    def wrapper (*args, **kwargs):

        # Get result of base transform
        result = tf(*args, **kwargs)

        # Check
        assert len(result.shape) == 1 and result.dtype.names is not None

        # Regularise output
        return result.astype([(feat, '<f8') for feat in result.dtype.names]).view('<f8').reshape((result.shape[0], len(result.dtype.names)))

    return wrapper


@regularise
def tf_ATLAS (batch):
    """
    Select only features necessary to evaluate ATLAS PID performance.

    Arguments:
        batch: numpy recarray read from HDF5 file

    Returns:
        Regular numpy array containing only specified fields.
    """

    # Specify necessary features
    feats = ['LHValue']
    feats = ['probe_{}'.format(feat) for feat in feats]

    return batch[feats]


@regularise
def tf_flat (batch):
    """
    Select only features necessary to train/evaluate vanilla NN or similar.

    Arguments:
        batch: numpy recarray read from HDF5 file

    Returns:
        Regular numpy array containing only specified fields.
    """

    # Specify necessary features
    feats = ['Rhad1', 'Rhad', 'f3', 'weta2', 'Rphi', 'Reta', 'Eratio', 'f1']
    feats = ['probe_{}'.format(feat) for feat in feats]

    return batch[feats]


def tf_images (batch):
    """
    Select only features necessary to train/evaluate CNNs.

    Arguments:
        batch: numpy recarray read from HDF5 file

    Returns:
        ...
    """

    # Specify necessary features
    feats = ['layer{:d}'.format(i) for i in range(4)]
    feats = ['image_{}'.format(feat) for feat in feats]

    images = [batch[feat] for feat in feats]
    return [np.expand_dims(image, axis=-1) for image in images]


# Disable direct imports
__all__ = []

# Create struct-type which is only instance to be imported
TransformsStruct = namedtuple("TransformsStruct", "ATLAS flat images")
transforms = TransformsStruct(ATLAS=tf_ATLAS, flat=tf_flat, images=tf_images)
