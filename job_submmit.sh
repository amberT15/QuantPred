#!/bin/bash
source ~/.bash_profile

NAME=bpnet_layer_search
DIR_CODE=/grid/koo/home/ztang/QuantPred/
d=`date +%y-%m-%d-%H-%M-%S`
DIR_LOG=/grid/koo/home/ztang/quantlog/$NAME/$d
SWEEP_ID=ambert/bpnet_layer_search/sweeps/phof2fkw
export SWEEP_ID

#$ -N bpnet_layer_search
#$ -l m_mem_free=3G
#$ -l gpu=1
#$ -e /grid/koo/home/ztang/quantlog/log
#$ -o /grid/koo/home/ztang/quantlog/log

cd $DIR_CODE
singularity exec --nv quantpred_latest.sif python wandb_train.py $SWEEP_ID 1
