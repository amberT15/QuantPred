#!/bin/sh
dataset1=HepG2
dataset2=K562

./filter_encode_data.py ./datasets/QQ_encode_TF.txt datasets/$dataset1 -b $dataset1 --no_crispr -l 7

./filter_encode_data.py ./datasets/QQ_encode_TF.txt datasets/$dataset2 -b $dataset2 --no_crispr -l 7


./download_urls.py ./datasets/${dataset1}/urls.txt ./datasets/${dataset1}
./download_urls.py ./datasets/${dataset2}/urls.txt ./datasets/${dataset2}
