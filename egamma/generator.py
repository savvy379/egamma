# -*- coding: utf-8 -*-

"""
Utilities for multiprocess loading of data files.
"""

# Basic import(s)
import os
import time
import h5py
import logging as log
import itertools
import multiprocessing
from subprocess import call

log.basicConfig(format='[%(levelname)s] %(message)s', level=log.INFO)


def cautious_remove (path):
    """
    ...
    """
    if path.startswith('/') or '*' in path:
        log.info("cautious_remove: Refusing to remove {}".format(path))
    else:
        log.debug("cautious_remove: Removing.")
        call(['rm', path])
        pass
    pass


def unique_tmp (path):
    """
    Utility script to create a unique, temporary file path.
    """
    ID = int(time.time() * 1E+06)
    basedir = '/'.join(path.split('/')[:-1])
    filename = path.split('/')[-1]
    filename = 'tmp.{:s}.{:d}'.format(filename, ID)
    return '{}/{}'.format(basedir, filename)


class DecompressionProcess (multiprocessing.Process):

    def __init__(self, paths, queue):
        # Base class constructor
        multiprocessing.Process.__init__(self)

        # Member variable(s)
        self.__paths = paths
        self.__queue = queue

        # Event(s)
        self.exit = multiprocessing.Event()
        self.done = multiprocessing.Event()
        pass


    def run (self):
        """
        ...
        """

        for path in itertools.cycle(self.__paths):
            log.debug("decompression: Decompressing {} ...".format(path))
            remove = False

            # Compressed with bzip2
            if path.endswith('.bz2'):
                path_hdf5 = path.replace('.bz2', '')
                path_hdf5 = unique_tmp(path_hdf5)
                assert not os.path.isfile(path_hdf5), \
                    "Trying to unzip {} but {} alrady exists.".format(path, path_hdf5)
                with open(path_hdf5, 'wb') as stdout:
                    call(['bzip2', '-dkc', path], stdout=stdout)
                    pass
                remove = True

            # Compressed with gzip
            elif path.endswith('.gz'):
                path_hdf5 = path.replace('.gz', '')
                path_hdf5 = unique_tmp(path_hdf5)
                assert not os.path.isfile(path_hdf5), \
                    "Trying to unzip {} but {} alrady exists.".format(path, path_hdf5)
                with open(path_hdf5, 'wb') as stdout:
                    call(['gunzip', '-c', path], stdout=stdout)
                    pass
                remove = True

            # Uncompressed
            elif path.endswith('.h5'):
                path_hdf5 = path

            # Unsupported
            else:
                log.warning("Path {} not supported.".format(path))
                continue

            # Add to queue
            done = False
            while not done:
                try:
                    self.__queue.put((path_hdf5, remove), timeout=1)
                    done = True
                except:
                    # Queue is fill
                    if self.exit.is_set():
                        done = True

                        # Clean-up
                        if remove:
                            cautious_remove(path_hdf5)
                            pass
                        pass
                    pass
                pass

            # Manual stop condition
            if self.exit.is_set():
                break
            pass

        self.done.set()
        return


    def stop (self):
        self.exit.set()
        return

    pass


class StagingProcess (multiprocessing.Process):

    def __init__(self, dataset_name, batch_size, queue_in, queue_out):
        # Base class constructor
        multiprocessing.Process.__init__(self)

        # Member variable(s)
        self.__dataset_name = dataset_name
        self.__batch_size   = batch_size
        self.__queue_in     = queue_in
        self.__queue_out    = queue_out

        # Event(s)
        self.exit = multiprocessing.Event()
        self.done = multiprocessing.Event()
        pass


    def run (self):
        """
        ...
        """

        while not self.exit.is_set():
            # Get opened, decompressed HDF5 file instance
            log.debug("staging: Waiting for input... ({})".format(self.__queue_in.qsize()))

            # Get the input path, if any
            path, remove = self.__queue_in.get()

            if path == None:
                break

            quit = False
            # Iterate through dataset
            with h5py.File(path) as hf:
                dataset = hf[self.__dataset_name]
                idx = 0
                log.debug("staging: Start processing {}".format(path))
                while True:
                    batch = dataset[idx * self.__batch_size:(idx + 1)*self.__batch_size]
                    if batch.size == 0:
                        log.debug("Batch was empty. Stopping this file ({}).".format(path))
                        break
                    log.debug("staging: Putting batch from {} in __queue_stage".format(path))

                    # Add to queue
                    done = False
                    while not done:
                        try:
                            self.__queue_out.put(batch, timeout=1)
                            done = True
                        except:
                            # Queue is fill
                            if self.exit.is_set():
                                done = True
                                pass
                            pass
                        pass
                    idx += 1

                    # Manual stop condition
                    if self.exit.is_set():
                        break
                    pass

                log.debug("staging: Done processing {}".format(path, remove))
                if remove:
                    cautious_remove(path)
                    pass
                pass
            pass

        log.warning("Exiting StagingProcess!")
        self.done.set()
        return


    def stop (self):
        self.exit.set()
        return

    pass


class Generator:
    """
    Data staging generator with context manager structure.
    """

    def __init__ (self, paths, dataset_name, batch_size=1024, maxsize=8, num_batches=-1):
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
        self.__num_batches = num_batches

        self.__counter = 0
        self.__queue_stage      = multiprocessing.Queue(maxsize=maxsize)
        self.__queue_decompress = multiprocessing.Queue(maxsize=1)
        self.__process_decompress = DecompressionProcess(self.__paths, self.__queue_decompress)
        self.__process_stage      = StagingProcess(self.__dataset_name, self.__batch_size, self.__queue_decompress, self.__queue_stage)

        print "[INFO]  Starting generator with {} paths, batch size {} and max. queue size {}.".format(len(self.__paths), batch_size, maxsize)

        # Create processes
        #self.__process_decompress = Process(target=decompressing, args=(self.__paths, self.__queue_decompress))
        #self.__process_stage      = Process(target=staging, args=(self.__dataset_name, self.__batch_size, self.__queue_decompress, self.__queue_stage))

        # Start process threads
        self.__process_decompress.start()
        self.__process_stage     .start()
        pass


    def __iter__ (self):
        while True:
            yield self.next()


    def next (self):
        assert self.__process_stage is not None, \
            "[ERROR] The Generator was not properly initialised."
        batch = self.__queue_stage.get()
        if batch is None:
            raise StopIteration
        if self.__num_batches > 0 and self.__counter >= self.__num_batches:
            raise StopIteration
        self.__counter += 1
        return batch


    def stop (self):
        """
        ...
        """

        # Stop child processes
        self.__process_stage     .stop()
        self.__process_decompress.stop()

        log.info("Waiting for processes to finish running")
        while not (self.__process_stage.done.is_set() and self.__process_decompress.done.is_set()):
            time.sleep(1)
            pass

        # Retrieve remaining elements in queues
        paths   = list()
        batches = list()

        # -- Paths
        while self.__queue_decompress.qsize() > 0:
            paths.append(self.__queue_decompress.get())
            pass
        self.__queue_decompress.close()

        # -- Batches
        while self.__queue_stage.qsize() > 0:
            batches.append(self.__queue_stage.get())
            pass
        self.__queue_stage     .close()

        # Clean up any remaining decompressed files.
        if len(paths) > 0:
            for path, remove in paths:
                if remove:
                    cautious_remove(path)
                    pass
                pass
            pass

        # Wait for processes to finish
        self.__process_stage     .join()
        self.__process_decompress.join()
        return


    def __enter__ (self):
        return self


    def __exit__ (self, exc_type, exc_value, exc_traceback):
        self.stop()
        if exc_value != None:
            print 'Generator failed: %s' % (exc_value)
            pass
        return True

    pass
