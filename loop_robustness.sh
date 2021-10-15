#!/bin/bash

# qsub -q gpu_ded.q job_submmit.sh
for d in $1/* ; do
    echo "$d"
		qsub -q gpu_ded.q submit_robustness.sh $d
done
