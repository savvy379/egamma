# -*- coding: utf-8 -*-

"""
Utilities for multiprocess loading of data files.
"""

# Basic import(s)
import os
import time
import h5py
import numpy as np
import logging as log
import itertools
import multiprocessing
from subprocess import call

log.basicConfig(format='[%(levelname)s] %(message)s', level=log.INFO)

# Keras import(s)
import keras


# Define SplitArray class
class SplitArray (object):
    """docstring for SplitArray."""
    def __init__ (self, arrays, batch_size=1, endless=False):
        super(SplitArray, self).__init__()

        # Check(s)
        assert isinstance(arrays, list)

        # Member variable(s)
        self.batch_size = batch_size
        self.endless = endless

        self.arrays = arrays
        self.ranges = [(0,len(array) - 1) for array in arrays]
        for idx in range(1, len(self.arrays)):
            offset = self.ranges[idx - 1][1] + 1
            self.ranges[idx] = (self.ranges[idx][0] + offset, self.ranges[idx][1] + offset)
            pass
        return

    def __len__ (self):
        return sum(map(len, self.arrays)) // self.batch_size

    def __getitem__ (self, idx):

        # Slice accessor
        if isinstance(idx, slice):

            # Check(s)
            start = idx.start if idx.start is not None else  0
            stop  = idx.stop  if idx.stop  is not None else -1
            step  = idx.step  if idx.step  is not None else  1

            assert step == 1, "SplitArray: Reverse iteration is not supported."

            # Wrap ranges
            if start < 0:
                start = len(self) + start
                pass
            if stop < 0:
                stop = len(self) + stop
                pass

            # Iterate through slice
            output = list()
            try:
                for jdx in range(start, stop, step):
                    output.append(SplitArray.__getitem__(self, jdx))
                    pass
            except StopIteration:
                # Was out of range
                pass
            return output

        # Single-item accessor
        elif isinstance(idx, int):

            if self.endless:
                idx = idx % len(self)
            elif idx >= len(self):
                raise StopIteration

            # Find start- and stop indices for, and in, containers
            iarray_begin = map(lambda r: r[0] <=  idx      * self.batch_size <= r[1], self.ranges).index(True)
            iarray_end   = map(lambda r: r[0] <= (idx + 1) * self.batch_size <= r[1], self.ranges).index(True)
            ientry_begin =  idx      * self.batch_size - self.ranges[iarray_begin][0]
            ientry_end   = (idx + 1) * self.batch_size - self.ranges[iarray_end]  [0]

            # Within-array
            if iarray_begin == iarray_end:
                return self.arrays[iarray_begin][ientry_begin:ientry_end]

            # Across-arrays
            else:
                return np.concatenate((self.arrays[iarray_begin][ientry_begin:],
                                       self.arrays[iarray_end]  [:ientry_end]))
                pass

        # Not recognised
        else:
            raise ValueError("SplitArray.__getitem__: Received idx of unrecognised type {}.".format(type(idx)))

        return None
    pass


# Preprocessing transforms
def tf_ATLAS (batch):
    """
    ...
    """
    feats = ['LHValue']
    feats = ['probe_{}'.format(feat) for feat in feats]

    return batch[feats].astype([(feat, '<f8') for feat in feats]).view('<f8').reshape((batch.size, len(feats)))


def tf_flat (batch):
    """
    ...
    """
    feats = ['Rhad1', 'Rhad', 'f3', 'weta2', 'Rphi', 'Reta', 'Eratio', 'f1']
    feats = ['probe_{}'.format(feat) for feat in feats]

    return batch[feats].astype([(feat, '<f8') for feat in feats]).view('<f8').reshape((batch.size, len(feats)))


class Generator (keras.utils.Sequence, SplitArray):

    def __init__(self, paths, batch_size, endless=False, transform=lambda x:x):
        """
        ...
        """
        # Member variable(s)
        self.paths = paths
        self.transform = transform

        self.files    = [h5py.File(path, 'r') for path in self.paths]
        self.datasets = [hf['egamma'] for hf in self.files]

        # Base class constructors
        keras.utils.Sequence.__init__(self)
        SplitArray.__init__(self, self.datasets, batch_size=batch_size, endless=endless)

        print "Generator: Created with {} datasets".format(len(self.datasets))
        return

    def __len__ (self):
        return SplitArray.__len__(self)

    def __getitem__ (self, idx):
        return SplitArray.__getitem__(self, idx)

    def __del__(self):
        for f in self.files:
            f.close()
            pass
        return
    pass


class MixGenerator (keras.utils.Sequence):

    def __init__(self, *generators, **kwargs):
        """
        ...
        """
        accepted_keywords = ['return_class_label', 'shuffle']
        for kw in kwargs:
            if kw not in accepted_keywords:
                log.warning("MixGenerator: Keyword {} not accepted.".format(kw))
                kwargs.pop(kw)
                pass
            pass
        self.generators = generators

        # Keyword arguments
        self.return_class_label = kwargs.get('return_class_label', True)
        self.shuffle            = kwargs.get('shuffle',            True)
        return

    def __len__(self):
        return max(map(len, self.generators))

    def __getitem__(self, idx):

        # Combine parts from all generators
        parts  = [gen[idx] for gen in self.generators]
        labels = [np.ones((part.shape[0],)) * ipart for (ipart, part) in enumerate(parts)]
        batch = np.concatenate(parts)
        label = np.concatenate(labels)

        # Shuffle samples
        if self.shuffle:
            indices = np.arange(batch.shape[0])
            np.random.shuffle(indices)
            batch = batch[indices]
            label = label[indices]
            pass

        # Return batch and, optionally, class labels
        if self.return_class_label:
            return batch, label
        return batch

    pass
