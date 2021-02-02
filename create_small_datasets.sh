#!/bin/sh


home/shush/profile/QuantPred/filter_encode_data.py /home/shush/profile/QuantPred/datasets/QQ_encode_TF.txt datasets/ -b HepG2 --no_crispr -l 7

home/shush/profile/QuantPred/filter_encode_data.py /home/shush/profile/QuantPred/datasets/QQ_encode_TF.txt datasets/ -b K562 --no_crispr -l 7


/home/shush/profile/QuantPred/download_urls.py /home/shush/profile/QuantPred/datasets/HepG2_urls.txt /home/shush/profile/QuantPred/datasets/HepG2_bedfiles
/home/shush/profile/QuantPred/download_urls.py /home/shush/profile/QuantPred/datasets/K562_urls.txt /home/shush/profile/QuantPred/datasets/K562_bedfiles
