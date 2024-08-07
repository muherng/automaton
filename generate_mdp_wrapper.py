import sys
import os
# Add the parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import argparse
import itertools
import torch
from collections import Counter
import torch.nn.functional as F
import numpy as np
from generate_mdp import generate

#file is responsible for generating datasets 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--op",
        type=str,
        default="dfa",
        help="automaton type",
    )
    parser.add_argument("--train-batches", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2**10)
    parser.add_argument(
        "--preferred-dtype",
        type=str,
        default='int64',
        help="Use this dtype if possible (int64, object)"
    )
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--mode", type=str, default='random')
    parser.add_argument("--states",type=int,default=2)

    #args for scheduling 
    parser.add_argument("--task_id", type=int, default = 0)
    parser.add_argument("--num_tasks", type=int, default=1)
    parser.add_argument("--num_files", type=int, default=1)
    args = parser.parse_args()
    print('args: ', args)

    my_task_id = args.task_id
    num_tasks = args.num_tasks
    #list of dictionaries of arguments that are used to generate data
    data_args = []

    #curriculum dictionary
    #this is sort of bad -- no we just have 72 directories right now no tree
    for states in [25]: 
        for iterations in range(1,2):
            for discount in np.arange(0.5, 1.0, 0.1).tolist(): 
                args_dict = {'states':states,
                             'iterations':iterations,
                             'discount': discount}
                data_args.append(args_dict)
    
    # Assign indices to this process/task
    my_args = data_args[my_task_id:len(data_args):num_tasks]
    for i in range(len(my_args)):
        args_dict = my_args[i]
        print('args: ', args_dict)
        args.states = args_dict['states']
        args.iterations = args_dict['iterations']
        args.discount = args_dict['discount']
        generate(args)
    