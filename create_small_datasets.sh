#!/bin/sh

./filter_encode_data.py /home/shush/profile/QuantPred/datasets/QQ_encode_TF.txt datasets/ -b HepG2 --no_crispr

./filter_encode_data.py /home/shush/profile/QuantPred/datasets/QQ_encode_TF.txt datasets/ -b K562 --no_crispr
