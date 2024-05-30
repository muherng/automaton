#!/bin/bash

#use job triples [4,48,1]
#LLsub ./submit.sh [NODES,NPPN,NTPP]


# Loading the required module
source /etc/profile
module load anaconda/2023a
source activate test3

python -u generate_wrapper.py --task_id $LLSUB_RANK --num_tasks $LLSUB_SIZE
#python -u generate.py   