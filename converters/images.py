#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Basic import(s)
import re
import os
import h5py
import numpy as np
from subprocess import call

# ROOT import(s)
try:
    import ROOT
    import root_numpy
except ImportError:
    print "[WARN] ROOT and/or root_numpy are not installed. This might lead to problems."
    pass

# Project import(s)
from egamma.utils import *

# Command-line arguments parser
import argparse

parser = argparse.ArgumentParser(description="Convert HDF5 with cell information to ones with calorimater images.")

parser.add_argument('--stop', action='store', default=None, type=int,
                    help='Maximum number of events to read.')
parser.add_argument('--outdir', action='store', default="images", type=str,
                    help='Output directory.')
parser.add_argument('paths', type=str, nargs='+',
                    help='ROOT file(s) to be converted.')


# Main function definition.
def main ():

    # Parse command-line arguments.
    args = parser.parse_args()

    # Validate arguments
    if not args.paths:
        print "No ROOT files were specified."
        return

    if not args.outdir.endswith('/'):
        args.outdir += '/'
        pass

    args.paths = sorted(args.paths)

    # Loop input paths
    pattern = '.*\/data\_([\w]+\_)?[\d]{8}\.h5(\.bz2)?$'  # (.../)data_([tag]_)[01234567].h5(.bz2)
    for path in args.paths:
        print "== {}".format(path)

        # Check(s)
        assert os.path.isfile(path),    "File {} doesn't exist.".format(path)
        assert re.match(pattern, path), "File {} is not supported.".format(path)
        unzip = path.endswith('.bz2')
        
        # Unzip file before converting
        if unzip:
            path_hdf5 = path.replace('.bz2', '')
            assert not os.path.isfile(path_hdf5), \
                "Trying to unzip {} but {} alrady exists.".format(path, path_hdf5)
            print "   Unzipping {}.".format(path)

            call(['bzip2', '-dk', path])
            path = path_hdf5
            pass

        # Read data
        with h5py.File(path, 'r') as hf:
            array = hf['egamma'][:]
            pass
        
        # Convert to images
        data = convert_images(array, stop=args.stop)

        # Save as HDF5
        mkdir(args.outdir)
        filename = path.split('/')[-1].replace('data', 'images')
        print "   Saving to {}".format(args.outdir + filename)
        with h5py.File(args.outdir + filename, 'w') as hf:
            hf.create_dataset('egamma',  data=data, compression='gzip')
            pass

        # Compress
        #print "   Compressing"
        #call(['bzip2', '-f', args.outdir + filename])

        # Clean up decompressed file.
        if unzip:
            print "   Removing decompressed file {} again.".format(path_hdf5)
            call(['rm', path_hdf5])
            pass

        pass

    return


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
        (0.0031,    0.0245 * 4),  # ECAL barrel layer 1
        (0.025,     0.0245),      # ...           ... 2
        (0.025 * 4, 0.0245 * 4),  # ...
        ]
    dim = (0.20, 0.20)  # Storing (-dimx, dimx, -dimy, dimy) section of detector.
    ROOT.gROOT.SetBatch(True)

    # Loop samples
    images = [[] for _ in range(4)]
    for irow in range(stop):
        # Get arrays
        sampling = data[irow]['cells_sampling']
        eta      = data[irow]['cells_eta']
        energy   = data[irow]['cells_energy']
        phi      = data[irow]['cells_phi']

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
        m_eta = np.dot(eta, np.clip(energy, 0, None)) / np.sum(np.clip(energy, 0, None))
        m_phi = np.dot(phi, np.clip(energy, 0, None)) / np.sum(np.clip(energy, 0, None))

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
    indices, features = zip(*filter(lambda tup: not tup[1].startswith('cells_'), enumerate(data.dtype.names)))
    features = list(features)
    
    # Get images dimensions in (eta, phi)
    columns = [data[feat] for feat in features] + map(np.array, images)#
    image_dims = [col.shape[1:] for col in columns[-len(images):]]

    # Construct compund dtype
    names = features + ['image_layer{}'.format(ilayer) for ilayer in range(len(images))]
    formats = [data.dtype[idx].type for idx in indices] + ['({},{})float32'.format(*dims) for dims in image_dims]
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
