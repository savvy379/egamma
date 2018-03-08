# -*- coding: utf-8 -*-

"""
Common utility methods for egamma project.
"""

# Basic import(s)
import os

def mkdir (path):
    """
    Script to ensure that the directory at `path` exists.

    Arguments:
        path: String specifying path to directory to be created.
    """

    # Check mether  output directory exists
    if not os.path.exists(path):
        print "mdkir: Creating output directory:\n  {}".format(path)
        try:
            os.makedirs(path)
        except OSError:
            # Apparently, `path` already exists.
            pass
        pass

    return
