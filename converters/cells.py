#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Basic import(s)
import re
import h5py
import json
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

parser = argparse.ArgumentParser(description="Convert ROOT files with nested structure into flat HDF5 files.")

parser.add_argument('--tag', action='store', type=str, required=True,
                    help='Tag the data category (Zee, Wev, etc.).')
parser.add_argument('--stop', action='store', default=None, type=int,
                    help='Maximum number of events to read.')
parser.add_argument('--max-processes', action='store', default=10, type=int,
                    help='Maximum number of concurrent processes to use.')
parser.add_argument('--outdir', action='store', default="cells", type=str,
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
        log.error("No ROOT files were specified.")
        return

    if args.max_processes > 20:
        log.error("The requested number of processes ({}) is excessive. Exiting.".format(args.max_processes))
        return

    if args.stop is not None:
        args.stop = int(args.stop)
        pass

    if not args.outdir.endswith('/'):
        args.outdir += '/'
        pass

    if args.shuffle:
        raise NotImplemented()

    args.paths = sorted(args.paths)

    for path in args.paths:

        # Base candidate selection
        if args.tag == 'Zee':
            selection = "(p_truth_pdgId == 11 && p_truth_parent_pdgId == 23 && tag2_exists == 0)"
        else:
            selection = "(p_truth_parent_pdgId == 23 && tag2_exists == 0)"  # "(p_truth_eta > -1.5 && p_truth_eta < 1.5)"

        # Read numpy array from file.
        f = ROOT.TFile(path, 'READ')
        tree = f.Get('tree')

        # Split indices into batches
        N = min(1000000, tree.GetEntries())  # @TEMP
        index_edges = map(int, np.linspace(0, N, args.max_processes + 1, endpoint=True))
        index_ranges = zip(index_edges[:-1], index_edges[1:])

        # Start conversion process(es)
        pool = multiprocessing.Pool(processes=args.max_processes)
        results = pool.map(converter, [(path, start, stop, selection) for (start, stop) in index_ranges])

        # Concatenate data
        data = np.concatenate(results)
        print data.shape

        # Save as gzipped HDF5
        mkdir(args.outdir)
        filename = 'cells_{}.h5'.format(args.tag)
        log.debug("  Saving to {}".format(args.outdir + filename))
        with h5py.File(args.outdir + filename, 'w') as hf:
            hf.create_dataset('egamma',  data=data, chunks=(min(1024, data.shape[0]),), compression='gzip')
            pass
        call(['gzip', '-f', args.outdir + filename])
        pass

    return


def converter (arguments):
    """
    Process converting standard-format ROOT file to HDF5 file with cell
    content.

    Arguments:
        path: Path to the ROOT file to be converted.
        args: Namespace containing command-line arguments, to configure the
            reading and writing of files.
    """

    # Unpack arguments
    path, start, stop, selection = arguments

    # Read-in data from ROOT.TTree
    array = root_numpy.root2array(path, 'tree', start=start, stop=stop, selection=selection)

    # Convert to HDF5-ready format.
    data = convert_cells(array)

    # Store result in shared dict
    return data


def downcast_type (x):
    """
    Systematically check array dtype and downcast to float32/int16 as
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
        try:
            return h5py.special_dtype(vlen=downcast_type(x[0]))
        except:
            return h5py.special_dtype(vlen=np.float32)
    elif 'float' in dtype:
        return np.float32
    elif 'int'   in dtype:
        return np.int16
    elif 'bool'  in dtype:
        return np.bool_
    assert False, "Unknown data type {} / {}".format(dtype, type_)


def convert_cells (data):
    """
    Method to convert standard array to suitable format for classifier.

    Arguments:
        data: numpy array returned by root_numpy.tree2array, to be formatted.

    Returns:
        Flattened numpy recarray prepared for saving to HDF5 file.
    """

    # Load in variable names
    with open('share/variables.json', 'r') as f:
        var_dict = json.load(f)
        pass

    scalars = map(str, var_dict['scalars'])
    vectors = map(str, var_dict['vectors'])

    # Result containers
    candidates = list()
    variables  = list()

    # Dummy field
    v = scalars[0]
    for icand in range(len(data[v])):

        # Candidate dict
        d = dict()

        # Append scalars
        for key in scalars + vectors:

            # Store candidate data
            d[key] = data[key][icand]

            # Add variable name
            if key not in variables:
                variables.append(key)
            else:
                assert icand > 0, "Variable {0} ({0}) set for first candidate.".format(key)
                pass
            pass

        # Append candidate to output
        candidates.append(d)
        pass

    # Format output as numpy structured arrays.
    formats = [downcast_type(candidates[0][var]) for var in variables]
    for pair in zip(variables, formats):
        try:
            np.dtype([pair])
        except:
            print "Problem for {}".format(pair)
            pass
        pass
    dtype  = np.dtype(zip(variables, formats))
    output = np.array([tuple([d[var] for var in variables]) for d in candidates], dtype=dtype)

    return output

# Main function call.
if __name__ == '__main__':
    main()
    pass
