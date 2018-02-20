# -*- coding: utf-8 -*-

"""
Common utility methods for egamma project.
"""

# Basic import(s)
import os

# Keras import(s)
from keras.models import Model
from keras.layers import Input, BatchNormalization, Dense


def mkdir (path):
    """
    Script to ensure that the directory at `path` exists.

    Arguments:
        path: String specifying path to directory to be created.
    """

    # Check mether  output directory exists
    if not os.path.exists(path):
        print "mdkir: Creating output directory:\n  {}".format(path)
        try:
            os.makedirs(path)
        except OSError:
            # Apparently, `path` already exists.
            pass
        pass

    return


def classifier_model (num_features):
    """
    Factory method to create Keras classifier models.

    Arguments:
        num_features: Number of input features.

    Returns:
        Keras model.
    """

    # Input
    clf_input = Input(shape=(num_features,))

    # Feature standardisation
    l = BatchNormalization()(clf_input)

    # Layers(s)
    l = Dense(16, activation='relu')(l)
    l = Dense(16, activation='relu')(l)
    l = Dense(16, activation='relu')(l)

    # Output
    clf_output = Dense(1, activation='sigmoid')(l)

    return Model(inputs=clf_input, outputs=clf_output)
