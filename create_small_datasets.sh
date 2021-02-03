#!/bin/sh


./filter_encode_data.py ./datasets/QQ_encode_TF.txt datasets/ -b HepG2 --no_crispr -l 7

./filter_encode_data.py./datasets/QQ_encode_TF.txt datasets/ -b K562 --no_crispr -l 7


./download_urls.py ./datasets/HepG2_urls.txt HepG2_bedfiles
./download_urls.py ./datasets/K562_urls.txt K562_bedfiles
