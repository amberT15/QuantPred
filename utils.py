#!/usr/bin/env python

import sys
import os
from shutil import copyfile
from os.path import dirname

def make_directory(path):
    """Short summary.

    Parameters
    ----------
    path : Full path to the directory

    """

    if not os.path.isdir(path):
        os.mkdir(path)
        print("Making directory: " + path)
    else:
        print("Directory already exists!")

def get_parent(filepath):
    return dirname(filepath)
