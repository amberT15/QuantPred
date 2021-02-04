#!/bin/bash

python ./bed_generation.py -m 200 -s 1000 -o HepG2_small_dataset -c ./datasets/hg19.chrom.sizes ./datasets/HepG2/HepG2_sample_beds.txt

#python ./bed_generation.py -m 200 -s 1000 -o K562_small_dataset -c ./datasets/hg19.chrom.sizes ./datasets/K562_sample_beds.txt
