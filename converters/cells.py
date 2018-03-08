#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
        Process converting standard-format ROOT file to HDF5 file with cell 
        content.
        
        Arguments: 
            path: Path to the ROOT file to be converted.
            args: Namespace containing command-line arguments, to configure the
                reading and writing of files.
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
        
        # Base candidate selection
        selection = "(p_truth_eta > -1.5 && p_truth_eta < 1.5)"
        
        # Read numpy array from file.
        f = ROOT.TFile(self.__path, 'READ')
        t = f.Get('tree')
        array = root_numpy.tree2array(t, stop=self.__args.stop, selection=selection)
        
        # Convert to HDF5-ready format.
        data = convert_cells(array)

        # Get current file index, assuming regular naming convetion.
        index = int(re.search('\_([\d]+)\.myOutput', self.__path).group(1))  
        
        # Save as gzipped HDF5
        mkdir(self.__args.outdir)
        filename = 'cells_{}_{:08d}.h5'.format(self.__args.tag, index)
        log.debug("  Saving to {}".format(self.__args.outdir + filename))
        with h5py.File(self.__args.outdir + filename, 'w') as hf:
            hf.create_dataset('egamma',  data=data, chunks=(min(1024, data.shape[0]),), compression='gzip')
            pass
        call(['gzip', '-f', self.__args.outdir + filename])
        return

    pass
    

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


def convert_cells (data):
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
