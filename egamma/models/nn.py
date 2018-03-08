 # -*- coding: utf-8 -*-

 """
 Utilities for creating simple neural network models.
 """

 # Keras import(s)
 from keras.models import Model
 from keras.layers import Input, BatchNormalization, Dense


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
