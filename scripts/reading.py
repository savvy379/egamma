#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script for testing loading speeds of different file formats."""

# Import(s)
import os
import bz2
import time
import glob
import h5py
import pandas as pd
from multiprocessing import Pool, Process, Queue
from subprocess import call

# Local import(s)
from profile import Profile, profile


# Manual decompression
def decompressing (paths, queue):
    """
    ...
    """

    for path in paths:
        print "[DEBUG] decompression: Decompressing {} ...".format(path)
        remove = True
        if path.endswith('.bz2'):
            path_hdf5 = path.replace('.bz2', '')
            assert not os.path.isfile(path_hdf5), \
                "Trying to unzip {} but {} alrady exists.".format(path, path_hdf5)
            call(['bzip2', '-dk', path])
        elif path.endswith('.gz'):
            path_hdf5 = path.replace('.gz', '')
            assert not os.path.isfile(path_hdf5), \
                "Trying to unzip {} but {} alrady exists.".format(path, path_hdf5)
            with open(path_hdf5, 'wb') as stdout:
                call(['gunzip', '-c', path], stdout=stdout)
                pass
        elif path.endswith('.h5'):
            path_hdf5 = path
            remove = False
        else:
            print "[WARN]  Path {} not supported.".format(path)
            continue
        
        queue.put((path_hdf5, remove))
        pass
    queue.put((None, None))
    return

# Data loading thread
def staging (dataset_name, batch_size, queue_in, queue_out):
    """
    ...
    """
    while True:
        # Get opened, decompressed HDF5 file instance
        print "[DEBUG] staging: Waiting for input... ({})".format(queue_in.qsize())
        path, remove = queue_in.get()

        if path == None:
            break

        # Iterate through dataset
        dataset = h5py.File(path)[dataset_name]
        idx = 0
        print "[DEBUG] staging: Start processing {}".format(path)
        while True:
            batch = dataset[idx * batch_size:(idx + 1)*batch_size]
            if batch.size == 0:
                break
            queue_out.put(batch)
            idx += 1
            pass

        print "[DEBUG] staging: Done processing {}".format(path, remove)
        if remove:
            if path.startswith('/') or '*' in path:
                print "[DEBUG] staging: Refusing to remove {}".format(path)
            else:
                print "[DEBUG] staging: Removing."
                call(['rm', path])
                pass
            pass
        pass

    queue_out.put(None)
    return


class Generator:
    """
    Data staging generator with context manager structure.
    """

    def __init__ (self, paths, dataset_name, batch_size=1024, maxsize=8):
        """
        Constructor.

        Arguments:
            ...

        """

        # Check(s)
        # ...

        # Member variable(s)
        self.__paths = paths
        self.__dataset_name = dataset_name
        self.__batch_size = batch_size
        self.__queue_stage      = Queue(maxsize=maxsize)
        self.__queue_decompress = Queue(maxsize=1)
        self.__process_decompress = None
        self.__process_stage      = None
        pass


    def __iter__ (self):
        assert self.__process_stage is not None
        while True:
            batch = self.__queue_stage.get()
            if batch is None:
                raise StopIteration
            yield batch
            pass


    def __enter__ (self):
        """
        ...
        """

        # Create processes
        self.__process_decompress = Process(target=decompressing, args=(self.__paths, self.__queue_decompress))
        self.__process_stage      = Process(target=staging, args=(self.__dataset_name, self.__batch_size, self.__queue_decompress, self.__queue_stage))

        # Start process threads
        self.__process_decompress.start()
        self.__process_stage     .start()
        return self


    def __exit__ (self, *args):
        # Wait for processes to finish
        self.__process_decompress.join()
        self.__process_stage     .join()

        # Set to null
        self.__process_decompress = None
        self.__process_stage      = None
        return
    
    pass
    


# Main function definition.
@profile
def main ():

    # Test file path
    #path = "input/MC_SigBkgElectrons_2M.h5"
    path = "output/data_test_00000000.h5"
    dataset_name = 'egamma'


    # Read entire HDF5 file
    """
    with Profile("Read entire file"):
        with h5py.File(path, 'r') as hf:
            data_h5 = hf[dataset_name][:]
            pass
        pass
    #"""


    # Read batched HDF5 file
    with Profile("Read batched file"):
        directory = 'input/'
        paths = sorted(glob.glob('output/data_test_0000000*.h5.gz'))
        batch_size = 1024
        maxsize = 8
        
        with Generator(paths, dataset_name, batch_size, maxsize) as generator:
            with Profile("Looping"):
                for step, batch in enumerate(generator):
                    # Do someting...
                    pass
                pass
            pass
        pass


    return 0


if __name__ == '__main__':
    main()
    pass
