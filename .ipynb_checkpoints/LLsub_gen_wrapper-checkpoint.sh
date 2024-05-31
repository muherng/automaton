#!/bin/bash

#use job triples
#LLsub ./LLsub_gen_wrapper.sh [1,48,1]


# Loading the required module (change as necessary) 
source /etc/profile
module load anaconda/2023a
source activate test3

python -u generate_wrapper.py --task_id $LLSUB_RANK --num_tasks $LLSUB_SIZE