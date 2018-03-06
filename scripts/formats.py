#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script for testing loading speeds of different file formats."""

# Import(s)
import gzip
import time
import h5py
import pandas as pd

# Local import(s)
from profile import Profile, profile

# Main function definition.
@profile
def main ():
  
    # Test file path
    path = "input/MC_SigBkgElectrons_2M{}"


    # Read files
    # --------------------------------------------------------------------------
    with Profile("Read CSV"):
        # Pandas DataFrame
        df_csv = pd.read_csv(path.format('.csv'))
        pass

    # Remove additional index columns
    df_csv = df_csv.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)

    with Profile("Read HDF5"):
        with h5py.File(path.format('.h5'), 'r') as hf:
            # Numpy recarray
            data_h5 = hf['dataset'][:]
            pass
        pass    

    with Profile("Read HDF5 (gzipped)"):
        with h5py.File(path.format('.gzh5'), 'r') as hf:
            # Numpy recarray
            data_gzh5 = hf['dataset'][:]
            pass
        pass    

    with Profile("Read MsgPack"):
        # Pandas DataFrame
        df_msp = pd.read_msgpack(path.format('.msp'))
        pass


    # Write files
    # --------------------------------------------------------------------------
    """
    # Store test file as HDF
    with h5py.File(path.format('.h5'), 'w') as hf:
        hf.create_dataset('dataset',  data=df_csv.to_records(index=False))
        pass

    # Store test file as gzipped HDF5
    with h5py.File(path.format('.gzh5'), 'w') as hf:
        hf.create_dataset('dataset',  data=df_csv.to_records(index=False), compression='gzip')
        pass

    # Store test file as msp
    df_csv.to_msgpack(path.format('.msp'))
    #"""

    return 0


# Main function call.
if __name__ == '__main__':
    main()
    pass
