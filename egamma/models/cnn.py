 # -*- coding: utf-8 -*-

"""
Utilities for creating convolutional neural network models.
"""

# Keras import(s)
import keras
import keras.backend as K
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Dropout, Activation, Flatten, GRU, Concatenate, GlobalMaxPooling2D, GlobalAveragePooling2D, Reshape, Lambda, AveragePooling2D, Multiply
from keras.models import Model

# TensorFlow import(s)
import tensorflow as tf


def feature_extractor_ (input_img, num_convblocks, num_convlayers, dropout=0.2,
                        batchnorm=False, filter_size=(3,3), pool_size=(2,2),
                        num_filters=16, filter_multiplier=2,
                        name='feature_extractor'):
    """
    Extract image features using convolutional filters.

    Arguments:
        input_img: Image from which features should be extracted.
        num_convblocks: Number of convolutional blocks (before each max-pooling
            operation) to be used.
        num_convlayers: Number of convolutional layers within each block.
        dropout: Dropout-regularisation fraction.
        batchnorm: Whether to use batch normalisation between convolutional
            layers and the following activation unit.
        filter_size: Size of convolutional filters.
        pool_size: Size of max-pooling filters.
        num_filters: Number of convolutional filters to use in each layer of the
            first convolution block.
        filter_multiplier: The multiplier by which the number of convolitional
            filters should grow for each successive convolutional block.
        name: Name of the TensorFlow scope within which operations are placed.

    Returns:
        Flattened (global average-pooled) list of learned image features,
        concatenated with the average activation in the input image (to give
        overall scale).
    """

    # Construct network
    with tf.name_scope(name):

        # Input layer
        x = input_img

        # Convolutional layer groups
        for m in range(num_convblocks):
            for n in range(num_convlayers):
                x = Conv2D(num_filters, filter_size, padding='same')(x)
                if batchnorm:
                    x = BatchNormalization()(x)
                    pass
                x = Activation('relu')(x)
                x = Dropout(dropout)(x)
                pass
            x = MaxPooling2D(pool_size=pool_size, padding='same')(x)
            num_filters = int(num_filters * filter_multiplier)
            pass

         # Flatten
        x = Concatenate()([GlobalAveragePooling2D()(x),
                           GlobalAveragePooling2D()(input_img)])

        pass

    return x


def task_network_ (input_feats, num_categories, num_denselayers,
                   activation='sigmoid', dropout=0.2, batchnorm=False,
                   name='task_network'):
    """
    Task-specific network module.

    Arguments:
        input_feats: Vector of learned image features, from feature extractor.
        num_categories: Number of output nodes.
        num_denselayers: Number of fully-connected, hidden layers.
        activation: Activation on the output node(s).
        dropout: Dropout-regularisation fraction.
        batchnorm: Whether to use batch normalisation between hidden layers and
            the following activation unit.
        name: Name of the TensorFlow scope within which operations are placed.

    Returns:
        Learned, task-specific result(s).
    """

    # Construct network
    with tf.name_scope(name):

        # Flatten
        if isinstance(input_feats, (list, tuple)):
            x = Concatenate()(input_feats)
        else:
            x = input_feats
            pass

        # Dense, hidden layer groups
        for i in reversed(range(num_denselayers)):
            x = Dense(2**(2 + i) * num_categories)(x)
            if batchnorm:
                x = BatchNormalization()(x)
                pass
            x = Activation('relu')(x)
            x = Dropout(dropout)(x)
            pass

        # Output layer
        x = Dense(num_categories, activation=activation)(x)
        pass

    return x


def cnn_ (input_shapes, num_aux_features, name='egamma'):
    """
    Factory method for the full, CNN-based network model.

    Arguments:
        input_shapes: List of the shapes/dimensions of the input images (barrel
            ECAL layers).
        name: Name of the Keras model.

    Returns:
        Full, CNN-based model.
    """

    # Create separate inputs for each layer
    inputs = [Input(shape=input_shape, name='input_layer{}'.format(idx))
              for (idx, input_shape) in enumerate(input_shapes)]

    # Extract features from each layer
    features = [feature_extractor_(input, 1, 1, batchnorm=True, name='FEx_layer{}'.format(idx))
                for (idx, input) in enumerate(inputs)]

    # Auxiliary features
    if num_aux_features > 0:
        input_aux = Input(shape=(num_aux_features,), name='input_aux')
        inputs   += [input_aux]
        features_aux = task_network_(input_aux, 4, 2, activation='relu', batchnorm=True, name='aux_calo')
        features    += [features_aux]
        pass

    # Get task outputs
    outputs = [task_network_(features, 1, 1, 'sigmoid',  name='task_PID')]      # PID
               #task_network_(features, 1, 1, 'softplus', name='task_energy')]  # Energy regression
               # @TEMP: Energy regression is taken out for simplicity
    return Model(inputs, outputs, name=name)
