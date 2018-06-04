#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Basic import(s)
import re
import os
import h5py
import numpy as np
import logging as log
import multiprocessing
from subprocess import call

log.basicConfig(format='[%(levelname)s] %(message)s', level=log.INFO)

# ROOT import(s)
try:
    import ROOT
    import root_numpy
except ImportError:
    log.warning("ROOT and/or root_numpy are not installed. This might lead to problems.")
    pass

# Project import(s)
from egamma.utils import *

# Command-line arguments parser
import argparse

parser = argparse.ArgumentParser(description="Convert HDF5 with cell information to ones with calorimater images.")

parser.add_argument('--stop', action='store', default=None, type=int,
                    help='Maximum number of events to read.')
parser.add_argument('--max-processes', action='store', default=10, type=int,
                    help='Maximum number of concurrent processes to use.')
parser.add_argument('--outdir', action='store', default="images", type=str,
                    help='Output directory.')
parser.add_argument('paths', type=str, nargs='+',
                    help='HDF5 file(s) to be converted.')


# Main function definition.
def main ():

    # Parse command-line arguments.
    args = parser.parse_args()

    # Validate arguments
    if not args.paths:
        log.error("No HDF5 files were specified.")
        return

    if args.max_processes > 20:
        log.error("The requested number of processes ({}) is excessive. Exiting.".format(args.max_processes))
        return

    if not args.outdir.endswith('/'):
        args.outdir += '/'
        pass

    args.paths = sorted(args.paths)

    # Batch the paths to be converted so as to never occupy more than
    # `max_processes`.
    path_batches = map(list, np.array_split(args.paths, np.ceil(len(args.paths) / float(args.max_processes))))

    # Loop batches of paths
    for ibatch, path_batch in enumerate(path_batches):
        log.info("Batch {}/{} | Contains {} files".format(ibatch + 1, len(path_batches), len(path_batch)))

        # Convert files using multiprocessing
        processes = list()
        for path in path_batch:
            p = FileConverter(path, args)
            processes.append(p)
            pass

        # Start processes
        for p in processes:
            p.start()
            pass

        # Wait for conversion processes to finish
        for p in processes:
            p.join()
            pass
        pass

    return


class FileConverter (multiprocessing.Process):

    def __init__ (self, path, args):
        """
        ...
        """
        # Base class constructor
        super(FileConverter, self).__init__()

        # Member variable(s)
        self.__path = path
        self.__args = args
        return

    def run (self):
        # Printing
        log.debug("Converting {}".format(self.__path))
        if self.__args.stop is not None:
            log.debug("  Reading {} samples.".format(stop))
            pass

        # Check(s)
        #pattern = '.*\/cells\_([\w]+\_)?[\d]{8}\.h5(\.gz)?$'  # (.../)cells_([tag]_)[01234567].h5(.gz)
        pattern = '.*\/cells\_([\w]+)\.h5(\.gz)?$'  # (.../)cells_([tag]_)[01234567].h5(.gz)
        assert os.path.isfile(self.__path),    "File {} doesn't exist.".format(self.__path)
        assert re.match(pattern, self.__path), "File {} is not supported.".format(self.__path)

        remove = False

        # Compressed with bzip2
        if self.__path.endswith('.bz2'):
            path_hdf5 = self.__path.replace('.bz2', '')
            assert not os.path.isfile(path_hdf5), \
                "Trying to unzip {} but {} alrady exists.".format(self.__path, path_hdf5)
            with open(path_hdf5, 'wb') as stdout:
                call(['bzip2', '-dkc', self.__path], stdout=stdout)
                pass
            remove = True

        # Compressed with gzip
        elif self.__path.endswith('.gz'):
            path_hdf5 = self.__path.replace('.gz', '')
            assert not os.path.isfile(path_hdf5), \
                "Trying to unzip {} but {} alrady exists.".format(self.__path, path_hdf5)
            with open(path_hdf5, 'wb') as stdout:
                call(['gunzip', '-c', self.__path], stdout=stdout)
                pass
            remove = True

        # Uncompressed
        elif self.__path.endswith('.h5'):
            path_hdf5 = self.__path

        # Unsupported
        else:
            log.warning("Path {} not supported.".format(self.__path))
            return

        # Read data
        with h5py.File(path_hdf5, 'r') as hf:
            array  = hf['egamma'][:]
            pass

        # Convert to images
        data = convert_images(array, stop=self.__args.stop)

        # Save as HDF5
        mkdir(self.__args.outdir)
        filename = path_hdf5.split('/')[-1].replace('cells', 'images')
        log.info("  Saving to {}".format(self.__args.outdir + filename))
        with h5py.File(self.__args.outdir + filename, 'w') as hf:
            hf.create_dataset('egamma', data=data, compression='gzip')
            pass

        # Clean up decompressed file.
        if remove:
            cautious_remove(path_hdf5)
            pass

        return

    pass


def convert_images (data, stop=None):
    """
    Method to convert float cell-level array to calorimeter images.

    Arguments:
        data: numpy array stored in flat, cell-level HDF5 files.

    Returns:
        List of (eta, phi, E) calorimeter images, as numpy arrays.
    """

    stop = stop or data.shape[0]

    # Hard-coded cell dimensions in (eta, phi). *These are approximate at best!*
    bin_width = [
        (0.025,     0.1),         # PS
        (0.003125,  0.0245 * 4),  # ECAL barrel layer 1
        (0.025,     0.0245),      # ...           ... 2
        (0.025 * 2, 0.0245),      # ...
        ]
    dim = (0.20, 0.20)  # Storing (-dimx, dimx, -dimy, dimy) section of detector.
    ROOT.gROOT.SetBatch(True)

    # Loop samples
    images = [[] for _ in range(4)]
    for irow in range(stop):
        # Get arrays
        sampling = data[irow]['p_cell_sampling']
        eta      = data[irow]['p_cell_eta']
        energy   = data[irow]['p_cell_energy']
        phi      = data[irow]['p_cell_phi']

        # Number of cells
        N = energy.size

        # Ensure reasonable phi
        Phi  = phi.reshape((N,1)).repeat(N, axis=1)
        dPhi = np.abs(Phi - Phi.T)
        if N > 0 and np.max(dPhi) > np.pi:  # Cluster wrapping around phi = ±π
            phi[phi < 0] += 2. * np.pi
            phi -= np.pi
            pass

        # Get (eta,phi) barycenter
        eps = np.finfo(float).eps
        m_eta = np.dot(eta, np.clip(energy, 0, None)) / (np.sum(np.clip(energy, 0, None)) + eps)
        m_phi = np.dot(phi, np.clip(energy, 0, None)) / (np.sum(np.clip(energy, 0, None)) + eps)

        # Substract barycenter coordinates
        eta -= m_eta
        phi -= m_phi

        # Bin in per-layer histograms
        for ilayer in range(4):

            # Create histogram
            h = ROOT.TH2F('layer{}'.format(ilayer), "", int(2. * dim[0] / bin_width[ilayer][0]), -dim[0], dim[0],
                                                        int(2. * dim[1] / bin_width[ilayer][1]), -dim[1], dim[1])

            # Fill histogram
            msk_layer = (sampling == ilayer)
            root_numpy.fill_hist(h, np.vstack((eta,phi)).T[msk_layer,:], weights=np.clip(energy[msk_layer], 0, None))

            # Store array
            images[ilayer].append(root_numpy.hist2array(h))
            pass

        pass

    # Extract features to be propagated
    indices, features = zip(*filter(lambda tup: not tup[1].startswith('p_cell_') and not tup[1].startswith('tracks_'), enumerate(data.dtype.names)))
    features = list(features)

    # Get images dimensions in (eta, phi)
    columns = [data[feat] for feat in features] + map(np.array, images)
    image_dims = [col.shape[1:] for col in columns[-len(images):]]

    # Construct compund dtype
    names = features + ['image_layer{}'.format(ilayer) for ilayer in range(len(images))]
    formats = [data.dtype[idx].type for idx in indices] + ['({},{})float32'.format(*dims) for dims in image_dims]

    # -- @TEMP Filter out branches with `numpy.object_` type.
    good_indices = [ix for ix in range(len(formats)) if formats[ix] != np.object_]
    columns = [columns[ix] for ix in good_indices]
    names   = [names  [ix] for ix in good_indices]
    formats = [formats[ix] for ix in good_indices]

    dtype = np.dtype(zip(names, formats))

    # Construct samples a list of tuples
    samples = map(tuple, zip(*columns))

    # Format as numpy.recarray
    output = np.array(samples, dtype=dtype)

    return output


# Main function call.
if __name__ == '__main__':
    main()
    pass
