#!/bin/bash

#SBATCH -c 40
#SBATCH --gres=gpu:volta:2

#LLsub ./LLsub_train_wrapper.sh 

# Loading the required module
source /etc/profile
module load anaconda/2023a
source activate test3

#python -u train_wrapper_mdp.py --seq enc-dec
python -u train_wrapper_mdp.py --seq enc --kind hybrid --states 16