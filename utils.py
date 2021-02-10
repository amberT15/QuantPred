#!/usr/bin/env python
import urllib.request
import sys
import os
import pandas as pd
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


def download_metadata(metadata_url, output_folder):
    print('~~~~~~~~~~~~')
    print(output_folder)
    print("Downloading metadata.tsv for the project")
    metadata_path = os.path.join(output_folder, 'metadata.tsv')
    urllib.request.urlretrieve(metadata_url, metadata_path)
    metadata = pd.read_csv(metadata_path ,sep='\t')
    return metadata
