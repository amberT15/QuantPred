#!/bin/bash

./list_to_dataset.py
./bam_to_bw.sh datasets/A549/bam/ /home/shush/genomes/GRCh38_EBV.chrom.sizes.tsv datasets/A549/raw
