#!/usr/bin/env python
import sys
import os
from shutil import copyfile
import urllib.request
import utils
import subprocess

def main():
    urls_path = sys.argv[1] # filepath
    output_folder = sys.argv[2] # just foldername
    exp_folder = utils.get_parent(urls_path) # split parent dir
    urls_copy_path = os.path.join(exp_folder, "urls_copy.txt")
    output_dir = os.path.join(exp_folder, output_folder)
    utils.make_directory(output_dir)
    # look for existing copy file urls_copy.txt
    # set paths and copy url file if not already present
    if not os.path.isfile(urls_copy_path):
        # use this in the function that does the rest
        try:
            copyfile(urls_path, urls_copy_path)
        except:
            print('No urls.txt file provided!')

    process_file(urls_copy_path, output_dir)

# read paths
def process_file(urls_copy_path, output_path):
    with open(urls_copy_path, "r") as file:
        # read lines of the copy files
        urls = [f.strip() for f in file.readlines()]

    for url in urls:
        print("Downloading file "+url)
        one_output_path = os.path.join(output_path, url.split('/')[-1])
        # download url to the output dir
        cmd = 'wget -O {} {}'.format(one_output_path, url)
        subprocess.call(cmd, shell=True)
        # urllib.request.urlretrieve(url, output_path)
        # once completed delete the url line that was just now processed
        urls = urls[1:]
        with open(urls_copy_path, 'w') as f:
            for u in urls:
                f.write("%s\n" % u)
    # once all done delete the copy file
    os.remove(urls_copy_path)

if __name__ == '__main__':
  main()
