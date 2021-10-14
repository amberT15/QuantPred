#!/bin/bash
source ~/.bash_profile
i=1
while [ $i -le 20 ]
do
	qsub -q gpu_ded.q job_submmit.sh
	i=$(( $i+1 ))
done
