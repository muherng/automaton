import sys
import os

# Add the parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import argparse
import itertools
import torch
import tqdm
from tqdm import tqdm # For fancy progress bars
from collections import Counter
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import numpy as np
import sys
import os

#from generate import create_directory
#from train import training_step, validation_step
from model_aut import EncModel
from generate_aut import make_program

from seq2seq.src.model import Translator
#from seq2seq.new_main import train,validate 
from seq2seq.src.data import create_mask, generate_square_subsequent_mask # Loading data and data preprocessing

# Train on the GPU if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#file is responsible for training, saving models, and validation  
#python generate.py --task_id 0 --num_tasks 1 

#TODO: Change to accomodate encdec models. 
def train_on_data(args,path_to_pretrain = None):
    #dataset does not have data unless generate batch is called
    #generated data is of form [automaton]<word>=trace
    dataset = make_program(args)
    
    #load data 
    directory = 'data_aut'
    load=False

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
        #create default arguments for enc-dec
        defaults = {'enc_layers': 5,
                    'dec_layers': 5,
                    'embed_size': 512,
                    'attn_heads': 8,
                    'dim_feedforward': 512,
                    'dropout': 0.1,
                    'lr': 1e-4}
        
        model = Translator(
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
        print('path to pretrain: ', path_to_pretrain)
        state_dict = torch.load(path_to_pretrain)
        #print('state_dict: ', state_dict)
        #for name, param in state_dict.items():
        #    print(f"{name}: {param.size()}")
        model.load_state_dict(torch.load(path_to_pretrain))
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"The model has {num_params} parameters")
    
    if args.compile:
        model = torch.compile(model)
    
    dataloader_training(model, dataset, args, defaults=defaults, np_data=np_data)
    #TODO: integrate manual training into dataloader training
    #manual_training(model, dataset, args, np_data=np_data)
    return 


#TODO: redundancy with program dataset 
class EncDecDataset(Dataset):
    def __init__(self, src, tgt):
        self.src = src
        self.tgt = tgt

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        sample = {'src': self.src[idx], 'tgt': self.tgt[idx]}
        return sample

def load_data(np_data,args): 
    batch_size = args.batch_size
    train_batches = args.train_batches
    val_batches = args.val_batches 
    src = np_data['src']
    tgt = np_data['tgt']
    if src.shape[0] != tgt.shape[0]: 
        raise ValueError('Must be equal numbers of src and tgt')
    num_data = src.shape[0]
    if src.shape[0] < (train_batches + val_batches)*batch_size: 
        print('less data than train and val batches in arguments')
        train_src = src[:num_data - batch_size*val_batches,:]
        train_tgt = tgt[:num_data - batch_size*val_batches,:]
        
        val_src = src[num_data - batch_size*val_batches:,:]
        val_tgt = tgt[num_data - batch_size*val_batches:,:]
    else: 
        train_src = src[:train_batches*batch_size,:]
        train_tgt = tgt[:train_batches*batch_size,:]
        
        array = np.arange(train_batches*batch_size, num_data)
        # Select k random integers from the array
        random_indices = np.random.choice(array, batch_size*val_batches, replace=False)
        
        val_src = src[random_indices]
        val_tgt = tgt[random_indices]
        

    train_src = torch.from_numpy(train_src)
    train_tgt = torch.from_numpy(train_tgt)
    train_src = train_src.requires_grad_(False).clone().detach().long()
    train_tgt = train_tgt.requires_grad_(False).clone().detach().long()
    train_iterator = EncDecDataset(train_src,train_tgt)
    # Create the dataloader
    train_dl = DataLoader(train_iterator, batch_size = batch_size)
    
    val_src = torch.from_numpy(val_src)
    val_tgt = torch.from_numpy(val_tgt)
    val_src = val_src.requires_grad_(False).clone().detach().long()
    val_tgt = val_tgt.requires_grad_(False).clone().detach().long()
    val_iterator = EncDecDataset(val_src,val_tgt)
    # Create the dataloader
    val_dl = DataLoader(val_iterator, batch_size = batch_size)

    return train_dl, val_dl

def greedy_decode(model, src, tgt, src_mask, max_len, start_symbol, end_symbol):

    # Move to device
    src = src.to(device)
    src_mask = src_mask.to(device)
    tgt = tgt.to(device)

    # Encode input
    memory = model.encode(torch.transpose(src,0,1), src_mask)
    _, batch_size, _ = memory.size()

    # Initialize the output tensor
    results = torch.zeros(tgt.shape).to(device)

    # Initialize the decoder input with start symbols
    ys = torch.ones(batch_size,1).fill_(start_symbol).type(torch.long).to(device)
    
    for token in range(tgt.shape[1]-1):
        # Create target mask
        tgt_mask = generate_square_subsequent_mask(ys.size(1), device).type(torch.bool).to(device)
        # Decode the encoded representation of the input
        out = model.decode(torch.transpose(ys,0,1), memory, tgt_mask)
        out = out.transpose(0, 1)

        # Convert to probabilities and take the max of these probabilities
        prob = model.ff(out[:, -1])
        _, next_words = torch.max(prob, dim=1)

        # Concatenate the new words to the ys tensor
        ys = torch.cat([ys, next_words.unsqueeze(1)], dim=1)

        # Update results tensor
        results[:, :ys.size(1)] = ys
    
    print('tgt: ', tgt[:10,:])
    print('results: ', results[:10,:])

    # Calculate accuracy
    acc = torch.sum(torch.all(results == tgt, dim=1)).item() / tgt.size(0)

    return results, acc

# Takes model and dataloader and computes fraction of examples where every token is predicted correctly
def inference(model,dl,special_symbols):

    # Get training data, tokenizer and vocab
    # objects as well as any special symbols we added to our dataset
    #_, _, src_vocab, tgt_vocab, src_transform, _, special_symbols = get_data(opts)
    # Set to inference
    model.eval()

    for batch in tqdm(dl): 
        src = batch['src']
        tgt = batch['tgt'] 
        #src = torch.transpose(src,0,1)
        #tgt = torch.transpose(tgt,0,1)
        tgt_length = tgt.shape[1]
        # Accept input and keep translating until they quit
        output_as_list = []
        # Convert to tokens
        num_tokens = src.shape[1]
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)

        # Decode
        _, acc = greedy_decode(
            model, src, tgt, src_mask, max_len=num_tokens*5, start_symbol=special_symbols["<bos>"], end_symbol=special_symbols["<eos>"]
        )
        
    return acc

# Train the model for 1 epoch
def train(model, chunk, loss_fn, optim, special_symbols):

    # Object for accumulating losses
    losses = 0
    # Put model into inference mode
    model.train()
    for batch in tqdm(chunk, ascii=True):
        print('progress bar?')
        src = batch['src']
        tgt = batch['tgt']
        src = src.to(device)
        tgt = tgt.to(device)
        
        print('src: ', src[:4,:])
        print('tgt: ', tgt[:4,:])   
        
        src = torch.transpose(src,0,1)
        tgt = torch.transpose(tgt,0,1)
        
        # We need to reshape the input slightly to fit into the transformer
        tgt_input = tgt[:-1, :]

        # Create masks
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, special_symbols["<pad>"], device)

        # Pass into model, get probability over the vocab out
        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)
        # Reset gradients before we try to compute the gradients over the loss
        optim.zero_grad()

        # Get original shape back
        tgt_out = tgt[1:, :]

        # Compute loss and gradient over that loss
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()
        # Step weights
        optim.step()
        # Accumulate a running loss for reporting
        losses += loss.item()

    # Return the average loss
    return losses / len(chunk)

# Check the model accuracy on the validation dataset
def validate(model, valid_dl, loss_fn, special_symbols):
    
    # Object for accumulating losses
    losses = 0

    # Turn off gradients a moment
    model.eval()

    for batch in tqdm(valid_dl, ascii=True):
        src = batch['src']
        tgt = batch['tgt']
        src = src.to(device)
        tgt = tgt.to(device)
        
        #transpose src and tgt to feed into model
        src = torch.transpose(src,0,1)
        tgt = torch.transpose(tgt,0,1)

        # We need to reshape the input slightly to fit into the transformer
        tgt_input = tgt[:-1, :]

        # Create masks
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, special_symbols["<pad>"], device)

        # Pass into model, get probability over the vocab out
        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        # Get original shape back, compute loss, accumulate that loss
        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    # Return the average loss
    return losses / len(list(valid_dl))

# Function to yield chunks of data
def chunked_dataloader(dataloader, chunk_size):
    """
    Generator to yield chunks of data from DataLoader.
    """
    chunk = []
    for batch in dataloader:
        chunk.append(batch)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk  # Yield the last chunk if it's not empty

def dataloader_training(model, dataset, args, defaults=None, np_data = None): 
    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = model.to(device)

    batch_size = args.batch_size
    #This line is only for enc models
    #optimizer = model.configure_optimizers()
    directory = 'saved_models'
    # Standard PyTorch Training Loop
    time_to_success = Counter()
    #validaiton accuracy array 
    val = []
    train_batches = args.train_batches
    
    special_symbols = dataset.special_symbols 
    
    if args.seq == 'enc-dec': 
        train_dl,valid_dl = load_data(np_data,args)

    # Set up our learning tools
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=special_symbols["<pad>"])

    # These special values are from the "Attention is all you need" paper
    optim = torch.optim.Adam(model.parameters(), lr=defaults['lr'], betas=(0.9, 0.98), eps=1e-9)

    best_val_loss = 1e6        
    for epoch in range(args.epochs):
        # Iterating through chunked DataLoader for train
        #validation dataloader is small so no need to chunk
        for chunk in chunked_dataloader(train_dl, chunk_size=10):
            print(f'Processing data chunk with {len(chunk)} batches')
            train_loss = train(model,chunk, loss_fn, optim, special_symbols)
            val_loss = validate(model, valid_dl, loss_fn, special_symbols)
            acc = inference(model,valid_dl,special_symbols)
            print('epoch: ', epoch)
            print('train loss: ', train_loss)
            print('validation loss: ', val_loss)
            print('acc: ', acc)
            val.append(acc)
            print("Validation: ", val)
            file = directory + f'/{args.seq}_{args.kind}s{args.states}w{args.word_length}_val.npy'
            np.save(file,val)

            file = directory + f'/{args.seq}_{args.kind}s{args.states}w{args.word_length}.pth' 
            torch.save(model.state_dict(), file)
            print('Saving Model') 

            if acc > 0.97: 
                file = directory + f'/{args.seq}_{args.kind}s{args.states}w{args.word_length}.pth' 
                print('Final Save: ', file)
                torch.save(model.state_dict(), file)
                return 

    return 



def manual_training(model, dataset, args, np_data=None):
    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = model.to(device)

    batch_size = args.batch_size
    optimizer = model.configure_optimizers()

    directory = 'saved_models'
    #TODO: Change to DataLoader
    
    # Standard PyTorch Training Loop
    time_to_success = Counter()
    #validaiton accuracy array 
    val = []
    for epoch in range(args.epochs):
        train_batches = args.train_batches
        with torch.no_grad():
            if np_data is None: 
                np_data = dataset.generate_batch(batch_size * train_batches)
            total_data = np_data

        print('np_data shape:', np_data.shape[0])
        if total_data.shape[0] > 2*batch_size: 
            train_data = total_data[:total_data.shape[0] - batch_size*args.val_batches*10,:]
            train_batches = int(train_data.shape[0]/batch_size)
            #train_batches = 10
            val_total = total_data[total_data.shape[0] - batch_size*args.val_batches*10:,:]
            random_indices = np.random.choice(val_total.shape[0], size=batch_size*args.val_batches, replace=False)
            val_data = val_total[random_indices]
            val_data = torch.tensor(val_data).to(device)
        else:
            raise ValueError('fewer than 2 times batch size total datapoints')
        # Training Loop
        data_split = 100
        for interval in range(data_split): 
            low = interval*int(train_batches/data_split)
            high = (interval+1)*int(train_batches/data_split)
            #numpy for loading purposes, then torch for training
            train_data = total_data[low*batch_size:high*batch_size,:]
            train_data = torch.tensor(train_data).to(device)
            model.train()
            for batch_idx in tqdm.tqdm(range(high - low)):
                batch = train_data[batch_idx * batch_size : (batch_idx + 1) * batch_size]
                optimizer.zero_grad()
                loss = training_step(model, batch)
                loss.backward()
                optimizer.step()

            # Validation Loop
            accs = []
            model.eval()
            with torch.no_grad():
                #default is 1 
                val_batches = args.val_batches
                #np_data = dataset.generate_batch(batch_size * val_batches,mode=mode,sign=sign)
                #val_data = torch.tensor(np_data).to(device)
                for batch_idx in tqdm.tqdm(range(val_batches)):
                    batch = val_data[batch_idx * batch_size : (batch_idx + 1) * batch_size]
                    acc = validation_step(model, batch)
                    accs.append(acc)
            acc = torch.mean(torch.tensor(accs))
            print(f"Validation acc: {acc:.5}")
            #saving validation score 
            val.append(acc)
            print("Validation: ", val)
            file = directory + f'/{args.kind}s{args.states}w{args.word_length}_val.npy'
            np.save(file,val)
            
            if interval%5 == 0:
                file = directory + f'/{args.kind}s{args.states}w{args.word_length}.pth' 
                torch.save(model.state_dict(), file)
                #file = directory + f'/{args.kind}_val.npy' 
                #np.save(file,val)
                print('Saving Model') 

            if acc > 0.97 or epoch == args.epochs - 1: 
                file = directory + f'/{args.kind}s{args.states}w{args.word_length}.pth' 
                print('FINAL SAVE: ', file)
                torch.save(model.state_dict(), file)
                # Directory path
                # Access the state_dict and print the size of each weight tensor
                #for name, param in model.state_dict().items():
                #    print(f"{name}: {param.size()}")
                return 
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
    args = parser.parse_args()
    print('args: ', args)
    train_on_data(args)
