#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Basic import(s)
import re
import h5py
import json
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

parser = argparse.ArgumentParser(description="Convert ROOT files with nested structure into flat HDF5 files.")

parser.add_argument('--tag', action='store', type=str, required=True,
                    help='Tag the data category (Zee, Wev, etc.).')
parser.add_argument('--stop', action='store', default=None, type=int,
                    help='Maximum number of events to read.')
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
    
    args.paths = sorted(args.paths)

    # Base selection
    selection = ""  # E.g.: "(p_eta > -1.5 && p_eta < 1.5)"

    # Read data from all files.
    for ipath, path in enumerate(args.paths):
        print "== {}".format(path)
        if args.stop is not None:
            print "   Reading {} samples.".format(stop)
            pass

        # Read numpy array from file.
        f = ROOT.TFile(path, 'READ')
        t = f.Get('tree')
        array = root_numpy.tree2array(t, stop=args.stop, selection=selection)

        # Convert to HDF5-ready format.
        data = convert_flatten(array)

        # Filter NaN/inf rows.
        is_bad = lambda matrix: np.any(np.isnan(matrix) | np.isinf(matrix))
        for irow in range(data.shape[0]):
            if True in map(is_bad, data[irow].tolist()):
                print "Row {}/{} is bad!".format(irow, data.shape[0])
                pass
            pass

        # Save as gzipped HDF5
        mkdir(args.outdir)
        filename = 'data_{}_{:08d}.h5'.format(args.tag, ipath)
        print "   Saving to {}".format(args.outdir + filename)
        with h5py.File(args.outdir + filename, 'w') as hf:
            hf.create_dataset('egamma',  data=data, chunks=(1024,))
            pass

        pass

    return


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
        return h5py.special_dtype(vlen=downcast_type(x[0]))
    elif 'float' in dtype:
        return np.float32
    elif 'int'   in dtype:
        return np.int16
    elif 'bool'  in dtype:
        return np.bool_
    assert False, "Unknown data type {} / {}".format(dtype, type_)


def convert_flatten (data):
    """
    Method to convert standard array to suitable format for classifier.

    Arguments:
        data: numpy array returned by root_numpy.tree2array, to be formatted.

    Returns:
        Flattened numpy recarray prepared for saving to HDF5 file.
    """

    # Load in variable names
    with open('variables.json', 'r') as f:
        var_dict = json.load(f)
        pass

    scalars = var_dict['scalars']
    vectors = var_dict['vectors']

    # Define variable name transforms
    pattern_scalar_in  = '{}'
    pattern_scalar_out = '{}'

    pattern_vector_in  = 'pX_{}_Cells'
    pattern_vector_out = 'cells_{}'

    regex = re.compile('^p(X)?\_')   # Prettify probe variable names

    # Result containers
    candidates = list()
    variables  = list()

    # Dummy field
    v = pattern_scalar_in.format(scalars[0])
    for icand in range(len(data[v])):
        
        # Candidate dict
        d = dict()
        
        # Append scalars
        for key in scalars:

            # Get variables names
            var_in  = pattern_scalar_in .format(key)
            var_out = pattern_scalar_out.format(key)
            
            # @FIXME: Hacky, but trying to improve the naming scheme a bit.
            var_out = regex.sub('probe_', var_out)

            # Store candidate data
            d[var_out] = data[var_in][icand]
            
            # Add variable name
            if var_out not in variables:
                variables.append(var_out)
            else:
                assert icand > 0, "Variable {} ({}) set for first candidate.".format(var_out, var_in)
                pass
            pass
        
        # Append vectors
        for key in vectors:

            # Get variables names
            var_in  = pattern_vector_in .format(key)
            var_out = pattern_vector_out.format(key)

            # Store candidate data
            d[var_out] = data[var_in][icand]
            
            # Add variable name
            if var_out not in variables:
                variables.append(var_out)
            else:
                assert icand > 0, "Variable {} ({}) set for first candidate.".format(var_out, var_in)
                pass
            pass

        # Append candidate to output
        candidates.append(d)
        pass
    pass

    # Format output as numpy structured arrays.
    formats = [downcast_type(candidates[0][var]) for var in variables]
    dtype  = np.dtype(zip(variables, formats))
    output = np.array([tuple([d[var] for var in variables]) for d in candidates], dtype=dtype)

    return output

# Main function call.
if __name__ == '__main__':
    main()
    pass
