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

# Train on the GPU if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_square_subsequent_mask(size, device):
    mask = (torch.triu(torch.ones((size, size), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

# Create masks for input into model
def create_mask(src, tgt, pad_idx, device):

    # Get sequence length
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    # Generate the mask
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=device).type(torch.bool)

    # Overlay the mask over the original input
    src_padding_mask = (src == pad_idx).transpose(0, 1)
    tgt_padding_mask = (tgt == pad_idx).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

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
    
class EncDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample

def seq2seq_load(np_data,args): 
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
        val_size = val_batches*batch_size
        # Generate a random permutation of indices
        indices = np.random.permutation(src.shape[0])

        # Split indices into training and validation sets
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]
        
        train_src = src[train_indices]
        train_tgt = tgt[train_indices]
        
        val_src = src[val_indices]
        val_tgt = tgt[val_indices]
    else: 
        train_src = src[:train_batches*batch_size,:]
        train_tgt = tgt[:train_batches*batch_size,:]
        
        array = np.arange(train_batches*batch_size, num_data)
        # Select k random integers from the array
        random_indices = np.random.choice(array, batch_size*val_batches, replace=False)
        
        val_src = src[random_indices]
        val_tgt = tgt[random_indices]
        

    print('device data: ', device)
    train_src = torch.from_numpy(train_src).to(device)
    train_tgt = torch.from_numpy(train_tgt).to(device)
    train_src = train_src.requires_grad_(False).clone().detach().long()
    train_tgt = train_tgt.requires_grad_(False).clone().detach().long()
    train_iterator = EncDecDataset(train_src,train_tgt)
    # Create the dataloader
    train_dl = DataLoader(train_iterator, batch_size = batch_size)
    
    val_src = torch.from_numpy(val_src).to(device)
    val_tgt = torch.from_numpy(val_tgt).to(device)
    val_src = val_src.requires_grad_(False).clone().detach().long()
    val_tgt = val_tgt.requires_grad_(False).clone().detach().long()
    val_iterator = EncDecDataset(val_src,val_tgt)
    # Create the dataloader
    val_dl = DataLoader(val_iterator, batch_size = batch_size)

    return train_dl, val_dl

def enc_load(np_data,args): 
    # Assuming np_data is your dataset and args contains batch_size and val_batches
    batch_size = args.batch_size
    val_batches = args.val_batches

    # Calculate the number of validation samples
    val_size = batch_size * val_batches

    # Generate a random permutation of indices
    indices = np.random.permutation(np_data.shape[0])

    # Split indices into training and validation sets
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]

    # Create training and validation datasets
    train_data = np_data[train_indices]
    val_data = np_data[val_indices]

    # Convert to PyTorch tensors
    print('device data: ', device)
    train_data = torch.tensor(train_data).to(device)
    val_data = torch.tensor(val_data).to(device)

    train_data = train_data.requires_grad_(False).clone().detach().long()
    train_iterator = EncDataset(train_data)
    # Create the dataloader
    train_dl = DataLoader(train_iterator, batch_size = batch_size)
    
    val_data = val_data.requires_grad_(False).clone().detach().long()
    val_iterator = EncDataset(val_data)
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
    
    #print('tgt: ', tgt[:10,:])
    #print('results: ', results[:10,:])

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
        # Accept input and keep translating until they quit
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
        src = batch['src']
        tgt = batch['tgt']
        src = src.to(device)
        tgt = tgt.to(device)
        
        #print('src: ', src[:4,:])
        #print('tgt: ', tgt[:4,:])   
        
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

def answer_mask(dataset, batch):
    """Creates a mask of everything after the EQ (or =) token, which separates the question
    from the answer."""
    mask = torch.cumsum(batch == dataset.eq_token, dim=1) == 1
    mask &= batch != dataset.eq_token
    return mask[:, 1:]

def train_enc(model, chunk, optim):
    # Object for accumulating losses
    losses = 0
    # Put model into inference mode
    model.train()
    for batch in tqdm(chunk, ascii=True):
        mask = answer_mask(model.ds, batch)
        truth = batch[:, 1:]
        #zero the gradients 
        optim.zero_grad()
        # Assuming 'model' is your model instance
        out = model(batch)[:, :-1]
        loss = F.cross_entropy(out[mask], truth[mask])
        loss.backward()
        # Step weights
        optim.step()
        # Accumulate a running loss for reporting
        losses += loss.item()

    # Return the average loss
    return losses / len(chunk)

def validate_enc(model, valid_dl):
    # Object for accumulating losses
    losses = 0

    # Turn off gradients a moment
    model.eval()

    for batch in tqdm(valid_dl, ascii=True):
        mask = answer_mask(model.ds, batch)
        truth = batch[:, 1:]
        out = model(batch)[:, :-1]
        loss = F.cross_entropy(out[mask], truth[mask])
        # Accumulate a running loss for reporting
        losses += loss.item()
    
    return losses/len(list(valid_dl))


def inference_enc(model, valid_dl):
    """Computes the accuracy on the model, if we assume greedy decoding is used.
    We only consider a question corectly solved if every single token is correctly predicted,
    including the padding."""
    # Turn off gradients a moment
    losses = 0
    model.eval()

    for batch in tqdm(valid_dl, ascii=True):
        mask = answer_mask(model.ds, batch)
        truth = batch[:, 1:]
        out = model(batch)[:, :-1]
        preds = torch.argmax(out, dim=2)

        # We'd to test that our validation method matches what you get with generate.
        # Unfortunately the LSTMs give slightly different results when passing a batch,
        # vs when passing one element at a time, which breaks the direct correspondance.
        for i in range(0):
            n = batch[i].tolist().index(model.ds.eq_token) + 1
            true = batch[i, n:]
            pred0 = preds[i, n - 1 :]
            pred1 = model.generate(batch[i][:n])
            if torch.all((preds * mask)[i] == (truth * mask)[i]):
                assert torch.all(pred0 == true)
                # If we are getting the answer right, they should be the same.
                assert torch.all(pred0 == pred1)
            else:
                # If we are getting the answer wrong, they should both be wrong.
                assert not torch.all(pred0 == true)
                assert not torch.all(pred1 == true)
            
        losses += torch.all(preds * mask == truth * mask, dim=1).float().mean()
    acc = losses / len(list(valid_dl))
    return acc.cpu().numpy()