# -*- coding: utf-8 -*-

"""
Utilities for cross-file loading of data.
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
from keras.utils import Sequence


# Define SplitArray class
class SplitArray (object):

    def __init__ (self, arrays, batch_size=1, endless=False):
        """
        Generic abstraction class which contains a collection of iterable containers,
        which it combines seamlessly to expose a single iterable container to the
        user. The base container class is generic, and requires only `__len__` and
        `__getitem__` methods; can by e.g. python list of h5py datasets.

        If the batch_size is greater than one, depending on the type of the base
        interable containers, there may be ambiguity in how to concatenate the parts
        of a slice split accross multiple files. Python lists and tuple, are
        concatenated as usual; otherwise `np.concatenate` is used.

        Arguments:
            arrays: List of iterable containers.
            batch_size: Number of items from the container to return on each
                `__getitem__` call.
            endless: Whether to wrap the index around the base containers, i.e.
                such that `sa[idx % len(sa)] == sa[idx]`.
        """
        super(SplitArray, self).__init__()

        # Check(s)
        assert isinstance(arrays, list)

        # Member variable(s)
        self.batch_size = batch_size
        self.endless = endless
        self.arrays = arrays
        self.ranges = [(0,len(array) - 1) for array in arrays]

        # Compute beginning- and end indices for each base array.
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
                output1 = self.arrays[iarray_begin][ientry_begin:]
                output2 = self.arrays[iarray_end]  [:ientry_end]

                if isinstance(output1, (list, tuple)):
                    return output1 + output2
                else:
                    return np.concatenate((output1, output2))
                pass

        # Not recognised
        else:
            raise ValueError("SplitArray.__getitem__: Received idx of unrecognised type {}.".format(type(idx)))

        return None
    pass


class HDF5Generator (Sequence, SplitArray):

    def __init__(self, paths, batch_size, endless=False, transform=lambda x:x):
        """
        Data generator class, based on `SplitArray`, suitable for training Keras
        models using `fit_generator`.

        Arguments:
            paths: List of paths to HDF5 files to iterate through.
            batch_size: See `SplitArray`.
            endless: See `SplitArray`.
            transform: The transform applied to batched read from the HDF5 files
                by `SplitArray`. Useful for formatting the data for a particular
                task (NN, CNN, etc.).
        """

        # Member variable(s)
        self.paths = paths
        self.transform = transform

        self.files    = [h5py.File(path, 'r') for path in self.paths]
        self.datasets = [hf['egamma'] for hf in self.files]

        # Base class constructors
        Sequence.__init__(self)
        SplitArray.__init__(self, self.datasets, batch_size=batch_size, endless=endless)

        print "Generator: Created with {} datasets".format(len(self.datasets))
        return

    def __len__ (self):
        return SplitArray.__len__(self)

    def __getitem__ (self, idx):
        return self.transform(SplitArray.__getitem__(self, idx))

    def __del__(self):
        for f in self.files:
            f.close()
            pass
        return
    pass


class MixGenerator (Sequence):

    def __init__(self, *generators, **kwargs):
        """
        Generator class, combining the outputs from multiple generators,
        allowing for concurrent reading different data sources, sample classes,
        etc. Notice that the data class labels returns by the `MixGenerator` are
        labelled such that the first generator corresponds to class `0`; the
        second generator to class `1`; etc. Therefore, it is recommended to use
        the [background, signal] ordering.

        Arguments:
            generators: List of generators to iterate through concurrently.
            kwargs: Keyword arguments. Allowed values are:
                return_class_label: Whether to return class label (see above) in
                    addition to data. Default is `True`.
                shuffle: Whether to shuffle data (and labels) befure returning.
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
        
        # Check output from generator(s)
        returns_list = isinstance(parts[0], (list, tuple))

        # Output is list of numpy arrays:
        if returns_list:
            labels = [np.ones((part.shape[0],)) * ipart for (ipart, part) in enumerate(parts[0])]
            batch = [np.concatenate(tuple(group), axis=0) for group in zip(*parts)]
        else:
            labels = [np.ones((part.shape[0],)) * ipart for (ipart, part) in enumerate(parts)]
            batch = np.concatenate(parts)
            pass
        label = np.concatenate(labels)

        # Shuffle samples
        if self.shuffle:
            if returns_list:
                indices = np.arange(batch[0].shape[0])
            else:
                indices = np.arange(batch.shape[0])
                pass

            np.random.shuffle(indices)

            label = label[indices]
            if returns_list:
                batch = [group[indices] for group in batch]
            else:
                batch = batch[indices]
                pass
            pass

        # Return batch and, optionally, class labels
        if self.return_class_label:
            return batch, label
        return batch

    pass
