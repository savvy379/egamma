#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script training CNNs for electron PID and energy regression."""


# Basic import(s)
import os
import sys
import time
import h5py
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from itertools import izip

# Set Keras backend
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress annoying TensorFlow msgs.

# PyTorch import(s)
import torch

# TensorFlow import(s)
if torch.cuda.is_available():
    import tensorflow

    # -- Manually configure Tensorflow session
    gpu_options = tensorflow.GPUOptions(per_process_gpu_memory_fraction=0.1,
                                allow_growth=True)

    config = tensorflow.ConfigProto(allow_soft_placement=True,
                            gpu_options=gpu_options)

    session = tensorflow.Session(config=config)

    # -- Import Keras backend
    import keras.backend as K
    K.set_floatx('float32')

    # -- Set global Tensorflow session
    K.set_session(session)
    pass

# Keras import(s)
import keras
if torch.cuda.is_available():
    from keras.utils import multi_gpu_model
    pass
from keras.utils.vis_utils import plot_model

# Project import(s)
# -- Add module path
module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)
    pass

from egamma.utils import *
from egamma.profile import *
from egamma.generator import *
from egamma.models.cnn import *
import egamma.transforms as tf


# Main function definition.
def main ():

    # Define variables
    num_gpus = 1  # Specify integer > 1 to run in parallel on GPUs

    batch_size      =  128
    steps_per_epoch = 1000
    epochs          =   50

    pattern = '{grp}/{grp}_{cls}*_00000*.h5'

    # Get data file paths
    paths_sig = sorted(glob(pattern.format(grp='images', cls='Z')))[:1]
    paths_bkg = sorted(glob(pattern.format(grp='images', cls='W')))[:1]

    # Create data generators for each class
    gen_train_sig = HDF5Generator(paths_sig, batch_size=batch_size // 2, transform=tf.images, endless=True)
    gen_train_bkg = HDF5Generator(paths_bkg, batch_size=batch_size // 2, transform=tf.images, endless=True)

    # Create common generator
    gen_train = MixGenerator(gen_train_bkg, gen_train_sig)

    # Get single training example
    X, y = gen_train[0]
    print [x.shape for x in X], y.shape

    # Create network model
    cnn = cnn_([x.shape[1:] for x in X], 0)

    # Plot model diagram
    mkdir('figures/')
    plot_model(cnn, to_file='figures/cnn.png', show_shapes=True)

    # Parallelise model
    if torch.cuda.is_available() and num_gpus > 1:
        parallelised = multi_gpu_model(cnn, num_gpus)
    else:
        parallelised = cnn
        pass

    # Compile model
    parallelised.compile(loss='binary_crossentropy', optimizer='adam')

    # Fit model
    parallelised.fit_generator(gen_train, steps_per_epoch=steps_per_epoch, epochs=epochs)

    # Save model
    mkdir('models/')
    cnn.save('models/cnn.h5')
    return 0


# Main function call
if __name__ == '__main__':
    main()
    pass
