#!/bin/bash
source ~/.bash_profile

DIR_CODE=/grid/koo/home/toneyan/profile/QuantPred
run_dir=$1
export run_dir

#$ -N submit_train
#$ -l m_mem_free=3G
#$ -l gpu=1
#$ -e /grid/koo/home/toneyan/profile/QuantPred/logs
#$ -o /grid/koo/home/toneyan/profile/QuantPred/logs
###

cd $DIR_CODE
export SGE_TASK_ID
singularity exec --nv quantpred_latest.sif python robustness_test_pipeline.py $run_dir
