import sys
import os

# Add the parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import argparse
import itertools
import torch
import torch.nn.functional as F
import numpy as np
import gc

#from generate import create_directory
#from train import training_step, validation_step
from models.enc_model import EncModel
from generate_aut import make_program
from train_helper import seq2seq_load, enc_load, inference, train, validate, chunked_dataloader, validate_enc, train_enc, inference_enc

from models.seq2seq_model import EncDecModel

# Train on the GPU if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_on_data(args,path_to_pretrain = None):
    #dataset does not have data unless generate batch is called
    #generated data is of form [automaton]<word>=trace
    #make program constructor does not generate actual tokens unless .generate is called
    dataset = make_program(args)
    
    #load data 
    directory = 'data_aut'

    if args.seq == 'enc': 
        np_data = np.load(directory+f'/s{args.states}w{args.word_length}.npy',mmap_mode='r')
        model = EncModel(
            ds=dataset,
            kind=args.kind,
            hidden_size=args.hidden_size,
            ffw_size=2 * args.hidden_size if args.ffw_size is None else args.ffw_size,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            lr=args.lr,
            dropout=args.dropout,
        )
    elif args.seq == 'enc-dec':
        src_np = np.load(directory+f'/src_s{args.states}w{args.word_length}.npy',mmap_mode='r')
        tgt_np = np.load(directory+f'/tgt_s{args.states}w{args.word_length}.npy',mmap_mode='r')
        np_data = {'src':src_np, 'tgt':tgt_np}
        # Create model
        #create default arguments for enc-dec from Vaswani et. al. 
        defaults = {'enc_layers': 5,
                    'dec_layers': 5,
                    'embed_size': 512,
                    'attn_heads': 8,
                    'dim_feedforward': 512,
                    'dropout': 0.1,
                    'lr': 1e-4}
        print('default settings: ', defaults)
        model = EncDecModel(
            num_encoder_layers= defaults['enc_layers'],
            num_decoder_layers=defaults['dec_layers'],
            embed_size=defaults['embed_size'],
            num_heads=defaults['attn_heads'],
            src_vocab_size=dataset.n_tokens,
            tgt_vocab_size=dataset.n_tokens,
            dim_feedforward=defaults['dim_feedforward'],
            dropout=defaults['dropout']
        )
    else: 
        raise NotImplementedError
    
    if path_to_pretrain is not None: 
        print('path to pretrained model: ', path_to_pretrain)
        #state_dict = torch.load(path_to_pretrain)
        #print('state_dict: ', state_dict)
        #for name, param in state_dict.items():
        #    print(f"{name}: {param.size()}")
        model.load_state_dict(torch.load(path_to_pretrain))
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"The model has {num_params} parameters")
    
    if args.compile:
        model = torch.compile(model)
    
    if args.seq == 'enc-dec': 
        seq2seq_training(model, dataset, args, defaults=defaults, np_data=np_data)
        del model, dataset, np_data, src_np, tgt_np
    if args.seq == 'enc': 
        enc_training(model, dataset, args, np_data=np_data)
        del model, dataset, np_data
    gc.collect() 
    return 

def seq2seq_training(model, dataset, args, defaults=None, np_data = None): 
    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = model.to(device)

    directory = 'saved_models'
    # Standard PyTorch Training Loop
    #validation accuracy array 
    acc_list = []
    
    special_symbols = dataset.special_symbols 
    
    if args.seq == 'enc-dec': 
        train_dl,valid_dl = seq2seq_load(np_data,args)

    # Set up our learning tools
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=special_symbols["<pad>"])

    # These special values are from the "Attention is all you need" paper
    optim = torch.optim.Adam(model.parameters(), lr=defaults['lr'], betas=(0.9, 0.98), eps=1e-9)
      
    for epoch in range(args.epochs):
        # Iterating through chunked DataLoader for train
        #validation dataloader is small so no need to chunk
        for chunk in chunked_dataloader(train_dl, chunk_size=50):
            print(f'Processing data chunk with {len(chunk)} batches')
            train_loss = train(model, chunk, loss_fn, optim, special_symbols)
            val_loss = validate(model, valid_dl, loss_fn, special_symbols)
            acc = inference(model,valid_dl,special_symbols)
            print('train loss: ', train_loss)
            print('validation loss: ', val_loss)
            print('acc: ', acc)
            acc_list.append(acc)
            print("Accuracy: ", acc_list)
            file = directory + f'/{args.seq}_{args.kind}s{args.states}w{args.word_length}_acc.npy'
            np.save(file,acc_list)

            file = directory + f'/{args.seq}_{args.kind}s{args.states}w{args.word_length}.pth' 
            torch.save(model.state_dict(), file)
            print('Saving Model') 

            if acc > 0.97: 
                file = directory + f'/{args.seq}_{args.kind}s{args.states}w{args.word_length}.pth' 
                print('Final Save: ', file)
                torch.save(model.state_dict(), file)
                torch.cuda.empty_cache()
                del model, optim, train_dl, valid_dl, np_data
                gc.collect()
                return
            
            torch.cuda.empty_cache()
    return 



def enc_training(model, dataset, args, np_data=None):
    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = model.to(device)
    #print('model device: ', device)

    # These special values are from the "Attention is all you need" paper
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    # Check optimizer parameter device

    directory = 'saved_models'
    
    # Standard PyTorch Training Loop
    #validaiton accuracy array 
    acc_list = []
    train_dl, valid_dl = enc_load(np_data,args)
    for epoch in range(args.epochs):
        for chunk in chunked_dataloader(train_dl, chunk_size=50):
            print(f'Processing data chunk with {len(chunk)} batches')
            train_loss = train_enc(model, chunk, optim)
            val_loss = validate_enc(model, valid_dl)
            acc = inference_enc(model,valid_dl)
            print('train loss: ', train_loss)
            print('validation loss: ', val_loss)
            print('acc: ', acc)
            acc_list.append(acc)
            print("Accuracy: ", acc_list)
            file = directory + f'/{args.seq}_{args.kind}s{args.states}w{args.word_length}_acc.npy'
            np.save(file,acc_list)

            file = directory + f'/{args.seq}_{args.kind}s{args.states}w{args.word_length}.pth' 
            torch.save(model.state_dict(), file)
            print('Saving Model') 

            if acc > 0.97: 
                file = directory + f'/{args.seq}_{args.kind}s{args.states}w{args.word_length}.pth' 
                print('Final Save: ', file)
                torch.save(model.state_dict(), file)
                return
            
            torch.cuda.empty_cache()
            #original plot is one accuracy computation every 50 batches 
    return

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
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
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
    args = parser.parse_args()
    print('args: ', args)
    train_on_data(args)
