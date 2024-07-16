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
import mdp as mdp

# Train on the GPU if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_directory(directory):
    if not os.path.exists(directory):
        # Create the directory
        os.makedirs(directory)
        print(f"Directory '{directory}' was created.")

def make_program(args):   
    kvargs = dict(
        states = args.states,
        iterations = args.iterations, 
        mode = args.mode,
        gamma = args.discount,
        preferred_dtype=args.preferred_dtype,
    )
    
    return mdp.grid_world(**kvargs)

def generate(args):
    dataset = make_program(args)
    print('Generating Program')
    batch_size = args.batch_size
    train_batches = args.train_batches
    #np_data = dataset.generate_batch(10)
    #np_data = dataset.generate_batch(10, mode=mode, sign=sign)
    # Directory path
    np_data = dataset.generate_batch(batch_size * train_batches)

    directory = 'data_aut'    
    discount = int(args.discount*10)
    file = directory + f'/s{args.states}i{args.iterations}d{discount}.npy'
    print('file: ', file)
        
    np.save(file, np_data)

    special_symbols = dataset.special_symbols
    save_data_to_src_tgt(np_data,special_symbols,args,directory)
    return 

def save_data_to_src_tgt(np_data,special_symbols,args,directory='data_aut'): 

    data = torch.tensor(np_data).to(device) 

    src,tgt =  split_src_tgt(data,special_symbols)

    discount = int(args.discount*10)
    np.save(directory + f'/src_s{args.states}i{args.iterations}d{discount}.npy', src)
    np.save(directory + f'/tgt_s{args.states}i{args.iterations}d{discount}.npy', tgt)

    return 

def split_src_tgt(data, special_symbols):
    # Determine the position of the '=' token in any row
    eq_token = special_symbols['<eq>']
    eq_pos = np.where(data[0] == eq_token)[0][0]

    # Determine the maximum lengths for src and tgt including <bos> and <eos>
    src_length = eq_pos + 1
    tgt_length = data.shape[1] - eq_pos + 1

    # Initialize src and tgt arrays
    src = np.full((data.shape[0], src_length), special_symbols['<pad>'], dtype=int)
    tgt = np.full((data.shape[0], tgt_length), special_symbols['<pad>'], dtype=int)

    # Fill the src array
    src[:, 0] = special_symbols['<bos>']
    src[:, 1:eq_pos] = data[:, 1:eq_pos]
    src[:, eq_pos] = special_symbols['<eos>']

    # Fill the tgt array
    tgt[:, 0] = special_symbols['<bos>']
    tgt[:, 1:] = data[:, eq_pos:]

    # Return the truncated src and tgt arrays
    return src, tgt

#load a numpy array from file and save to  
#TODO: implement function 
def load_save_to_src_tgt(file,special_symbols):
    np_data = np.load(file)
    #save_data_to_src_tgt(np_data,special_symbols,args,directory)

#file is responsible for generating datasets 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--op",
        type=str,
        default="mdp",
        help="automaton type",
    )
    parser.add_argument("--train-batches", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2**5)
    parser.add_argument(
        "--preferred-dtype",
        type=str,
        default='int64',
        help="Use this dtype if possible (int64, object)"
    )
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--mode", type=str, default='random')
    parser.add_argument("--states",type=int,default=4)
    parser.add_argument("--iterations",type=int,default=1)
    parser.add_argument("--discount",type=float,default=0.5)
    
    args = parser.parse_args()
    print('args: ', args)

    generate(args)

    data_args = []
    