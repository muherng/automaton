import argparse
import itertools
import torch
import tqdm
from collections import Counter
import torch.nn.functional as F
import numpy as np
import sys
import os
# Add the parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from train_aut import train_on_data
#file is responsible for training, saving models, and validation  
#python generate.py --task_id 0 --num_tasks 1 

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    #args for training model
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Number of examples to generate and train on",
    )
    parser.add_argument("--train-batches", type=int, default=1000)
    parser.add_argument("--val-batches", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam LR")
    parser.add_argument(
        "--acc-next", type=float, default=0.9, help="Accuracy before next level"
    )
    parser.add_argument("--dropout", type=float, default=0.01)
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=32,
        help="The hidden size for the neural network",
    )
    parser.add_argument(
        "--ffw-size",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=4,
        help="The number of layers for the neural network",
    )
    parser.add_argument("--batch-size", type=int, default=2**10, help="Batch size")
    parser.add_argument(
        "--kind",
        type=str,
        default='transformer',
        help="The type of neural network to use (lstm, transformer, hybrid)",
    )
    parser.add_argument(
        "--preferred-dtype",
        type=str,
        default='int64',
        help="Use this dtype if possible (int64, object)"
    )
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--num-heads",
        type=int,
        default=4,
        help="The number of heads/rank in transformer/mlp",
    )
    #args for generating data
    parser.add_argument(
        "--op",
        type=str,
        default="dfa",
        help="automaton type",
    )
    parser.add_argument("--seq", type=str, default="enc-dec",help="type of model enc-dec/enc/dec")
    parser.add_argument("--mode", type=str, default='random')
    parser.add_argument("--states",type=int,default=2)
    parser.add_argument("--alphabet",type=int,default=2)
    parser.add_argument("--word_length",type=int,default=1)

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
    for states in range(1,20): 
        for word_length in range(1,20):
            args_dict = {'states':states,'word_length':word_length}
            data_args.append(args_dict)
    
    # Assign indices to this process/task
    my_args = data_args[my_task_id:len(data_args):num_tasks]

    #TODO: create path to pretrained models
    path_to_pretrain = None
    
    for i in range(len(my_args)):
        args_dict = my_args[i]
        print('iteration: ', i)
        print('args: ', args_dict)
        args.states = args_dict['states']
        args.word_length = args_dict['word_length']
        train_on_data(args, path_to_pretrain)
        #after first iteration, we have pretrained models
        # Directory path
        directory = 'saved_models'
        #next iteration will use larger num args with this path to pretrain
        path_to_pretrain = directory + f'/{args.seq}_{args.kind}s{args.states}w{args.word_length}.pth' 