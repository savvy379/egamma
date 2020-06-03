 # -*- coding: utf-8 -*-

"""
Utilities for creating simple neural network models.
"""

# Keras import(s)
from keras.models import Model
from keras.layers import Input, BatchNormalization, Dense, Dropout


def nn_ (num_features, num_layers=3, num_nodes=16, num_outputs=1, activation_inner='relu', activation='sigmoid', batchnorm=False, dropout=None):
    """
    Factory method to create Keras classifier models.

    Arguments:
        num_features: Number of input features.
        num_layers: Number of (identical) hidden layers.
        num_nodes: Number of nodes per hidden layerself.
        num_outputs: Number of output nodes:
        activation_inner: Activation function to be applied to the outputs of
            the hidden layers.
        activation: Activation function to be applied to the output layer.
        batchnorm: Whether to use batch-normalisation after each hidden layer.
            Note, that batch-normalisation is always applied to the inputs, in
            order to ensure feature standardisation by defaultself.
        dropout: If not `None`, perform dropout regularisation after each hidden
            layer with probability `dropout`.

    Returns:
        Keras model.
    """

    # Input
    clf_input = Input(shape=(num_features,))

    # Feature standardisation
    l = BatchNormalization()(clf_input)

    # Layers(s)
    for _ in range(num_layers):
        l = Dense(num_nodes, activation=activation_inner)(l)
        if batchnorm:
            l = BatchNormalization()(l)
            pass
        if dropout is not None:
            l = Dropout(dropout)(l)
        pass

    # Output
    clf_output = Dense(num_outputs, activation=activation)(l)

    return Model(inputs=clf_input, outputs=clf_output)
