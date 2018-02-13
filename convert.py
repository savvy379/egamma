#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Basic import(s)
import h5py
import numpy as np

# ROOT import(s)
try:
    import ROOT
    import root_numpy
except ImportError:
    print "[WARN] ROOT and/or root_numpy are not installed. This might lead to problems."
    pass

# Project import(s)
from utils import *

# Command-line arguments parser
import argparse

parser = argparse.ArgumentParser(description="Convert ROOT files with nested structure into flat HDF5 files.")

parser.add_argument('--stop', action='store', default=None,
                    help='Maximum number of events to read.')
parser.add_argument('--split', action='store', default=1000,
                    help='Target number of candidates per file.')
parser.add_argument('--outdir', action='store', default="output", type=str,
                    help='Output directory.')
parser.add_argument('--shuffle', action='store_true', default=False,
                    help='Shuffle candidates before saving. (Requires reading entire dataset into memory.)')
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

    if args.stop is not None:
        args.stop = int(args.stop)
        pass

    if not args.outdir.endswith('/'):
        args.outdir += '/'
        pass

    # Read data from all files.
    for path in args.paths:
        # Read numpy array from file.
        f = ROOT.TFile(path, 'READ')
        t = f.Get('tree')
        array = root_numpy.tree2array(t, stop=args.stop)

        # Convert to HDF5-ready format.
        data_ = convert(array)

        # Concatenate to existing data array
        try:
            data = np.concatenate((data, data_))
        except NameError:
            data = data_
            pass
        print data.shape
        pass

    # Save as HDF5
    mkdir(args.outdir)
    if args.split > 0:
        idx = 0
        while idx * args.split < data.shape[0]:
            data_split = data[idx * args.split:(idx + 1) * args.split]
            with h5py.File(args.outdir + 'data_{:08d}.h5'.format(idx), 'w') as hf:
                hf.create_dataset('egamma',  data=data_split)
                pass
            idx += 1  # Increment counter
            pass
        pass
    else:
        with h5py.File(args.outdir + 'data.h5', 'w') as hf:
            hf.create_dataset('egamma',  data=data)
            pass
        pass

    return


def downcast_type (x):
    """
    Systematically check array dtype and downcast to float16/int16 as
    appropriate. Variable-length containers are downcast as
    `h5py.special_dtype` with the appropriate `vlen` type.

    Arguments:
        x: Numpy-type variable to be type-checked.

    Returns:
        Down-cast numpy/h5py dtype.
    """
    dtype = str(x.dtype)
    type_ = str(type(x))
    if 'ndarray' in type_:
        return h5py.special_dtype(vlen=downcast_type(x[0]))
    elif 'float' in dtype:
        return np.float16
    elif 'int'   in dtype:
        return np.int16
    elif 'bool'  in dtype:
        return np.bool_
    assert False, "Unknown data type {} / {}".format(dtype, type_)


def convert (data):
    """
    Method to convert standard array to suitable format for classifier.

    Arguments:
        data: numpy array return by root_numpy.tree2array, to be formatted

    Returns:
        Flatten numpy recarray prepared for saving to HDF5 file.
    """

    # Fields to read
    scalars = ['pt', 'et', 'eta', 'phi', 'type', 'origin', 'truthPdgId']                         # (num_cands,)
    vectors = ['energy', 'gain', 'provenance', 'sampling', 'time', 'x', 'y', 'z', 'eta', 'phi']  # (num_cands, num_cells)

    # Result containers
    candidates = list()
    names      = list()
    first = True

    # Dummy field
    v  = 'El_' + scalars[0]
    for ievent in range(len(data[v])):
        for icand in range(len(data[v][ievent])):

            # Candidate dict
            d = dict()

            # Append scalars
            for key_ in scalars:
                key  = 'El_{}'.format(key_)
                name = 'truth_' + key_.replace('truth', '')

                d[name] = data[key][ievent][icand]

                if first:
                    names.append(name)
                    pass
                pass

            # Append vectors
            for key_ in vectors:
                key  = 'El_{}_Cells'.format(key_)
                name = 'cells_' + key_

                d[name] = data[key][ievent][icand]

                if first:
                    names.append(name)
                    pass
                pass

            # Signal field
            name = 'signal'

            d[name] = np.abs(d['truth_PdgId']) == 11  # Truth electron

            if first:
                names.append(name)
                pass

            # Append to output
            candidates.append(d)

            # Set flag(s)
            first = False
            pass
        pass

    formats = [downcast_type(candidates[0][name]) for name in names]
    dtype  = np.dtype(zip(names, formats))
    output = np.array([tuple([d[name] for name in names]) for d in candidates], dtype=dtype)

    return output

# Main function call.
if __name__ == '__main__':
    main()
    pass
