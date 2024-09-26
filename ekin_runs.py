import math
import functools
import os
from typing import List, Optional

import numpy as np
import torch
import torch.optim as optim
from timm.scheduler import CosineLRScheduler as TimmCosineLRScheduler
from torch import nn
from tqdm import tqdm


# disable tqdm
# tqdm = lambda x: x

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_expression(P: torch.Tensor, Q: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
    """
    Compute the expression PZ(Z^TQZ) for given tensors P, Q, and Z.

    Args:
        P (torch.Tensor): Tensor of shape (m, d_out, d) (m is the head number).
        Q (torch.Tensor): Tensor of shape (m, d, d)
        Z (torch.Tensor): Tensor of shape (B, T, d)

    Returns:
        torch.Tensor: Resulting tensor of shape (B, T, d)
    """
    # Z: B x T x d
    # P: m x d x d
    # Q: m x d x d

    # Output: B x d x d
    B, T, d = Z.shape
    m, d_out, _ = P.shape

    # query
    Q = Q.reshape(m * d, d) # (md) x d
    QZ = Z @ Q.T # B x T x (md)
    QZ = QZ.reshape(B, T, m, d).transpose(1, 2) # B x m x T x d

    # value
    P = P.reshape(m * d_out, d) # (md) x d
    PZ = Z @ P.T # B x T x (md)
    PZ = PZ.reshape(B, T, m, d_out).transpose(1, 2) # B x m x T x d

    # key: normally we have another projection to produce multi head key
    Z = Z.unsqueeze(1) # B x 1 x T x  d
    Z = Z.repeat(1, m, 1, 1) # B x m x T x d
    # attention
    QZ = (QZ @ Z.transpose(-2, -1)) / math.sqrt(d) # B x m x T x T
     # Create a causal mask to ensure that each position can only attend to previous positions
    causal_mask = torch.tril(torch.ones(T, T, device=Z.device)).unsqueeze(0).unsqueeze(0)  # 1 x 1 x T x T
    QZ = QZ * causal_mask  # Apply the mask to the attention scores
    # output
    output = torch.matmul(QZ, PZ) # B x m x T x d

    return output.mean(dim=1) # B x T x d


class LinearAttention(nn.Module):
    """
    Linear Attention module that computes attention without softmax.
    """
    def __init__(self, d: int, d_out: int, n_head: int, mlp: bool, layer_norm: bool, winit: float = 1.0) -> None:
        """
        Initialize the LinearAttention module.

        Args:
            d (int): Input dimension.
            d_out (int): Output dimension.
            n_head (int): Number of attention heads.
            mlp (bool): Whether to use MLP.
            layer_norm (bool): Whether to use Layer Normalization.
            winit (float): Weight initialization factor.
        """
        super(LinearAttention, self).__init__()
        self.d = d
        self.d_out = d_out
        self.n_head = n_head
        self.mlp = mlp
        self.layer_norm = layer_norm
        self.P = nn.Parameter(torch.randn(n_head, d_out if not mlp else d, d) * winit / math.sqrt(d))
        self.Q = nn.Parameter(torch.randn(n_head, d, d) * winit / math.sqrt(d))
        self.layer_norm = nn.LayerNorm(d_out if not mlp else d) if layer_norm else None
        self.mlp = nn.Sequential(
            nn.Linear(d, 2*d),
            nn.ReLU(),
            nn.Linear(2*d, d)
        ) if mlp else None
        self.projection = nn.Linear(d, d_out) if  mlp else None

    def attention(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Compute the linear attention mechanism.

        Args:
            Z (torch.Tensor): Input tensor of shape (B, T, d).

        Returns:
            torch.Tensor: Output tensor after attention mechanism.
        """
        # Output: B x d x d
        B, T, d = Z.shape
        m, d_out, _ = self.P.shape

        # query
        Q = self.Q.reshape(m * d, d) # (md) x d
        QZ = Z @ Q.T # B x T x (md)
        QZ = QZ.reshape(B, T, m, d).transpose(1, 2) # B x m x T x d

        # value
        P = self.P.reshape(m * d_out, d) # (md) x d
        PZ = Z @ P.T # B x T x (md)
        PZ = PZ.reshape(B, T, m, d_out).transpose(1, 2) # B x m x T x d

        # key: normally we have another projection to produce multi head key
        Z = Z.unsqueeze(1) # B x 1 x T x  d
        Z = Z.repeat(1, m, 1, 1) # B x m x T x d

        # attention
        QZ = (QZ @ Z.transpose(-2, -1)) / math.sqrt(self.d) # B x m x T x T
        # Create a causal mask to ensure that each position can only attend to previous positions
        causal_mask = torch.tril(torch.ones(T, T, device=Z.device)).unsqueeze(0).unsqueeze(0)  # 1 x 1 x T x T
        QZ = QZ * causal_mask  # Apply the mask to the attention scores

        # output
        output = torch.matmul(QZ, PZ) # B x m x T x d

        y = output.mean(dim=1) # B x T x d

        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the LinearAttention module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying attention, layer norm, MLP, and projection.
        """
        residual = x

        x = self.attention(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x + residual)

        if self.mlp is not None:
            residual = x
            x = self.mlp(x)
            x = x + residual

        if self.projection is not None:
            x = self.projection(x)

        return x



class SoftmaxAttention(nn.Module):
    """
    Softmax Attention module that computes attention without softmax.
    """
    def __init__(self, d: int, d_out: int, n_head: int, mlp: bool, layer_norm: bool, winit: float = 1.0, have_output_proj=False, have_key_proj=False) -> None:
        """
        Initialize the SoftmaxAttention module.

        Args:
            d (int): Input dimension.
            d_out (int): Output dimension.
            n_head (int): Number of attention heads.
            mlp (bool): Whether to use MLP.
            layer_norm (bool): Whether to use Layer Normalization.
            winit (float): Weight initialization factor.
        """
        super(SoftmaxAttention, self).__init__()
        self.d = d
        self.d_out = d_out
        self.n_head = n_head
        self.mlp = mlp
        self.layer_norm = layer_norm
        self.have_output_proj = have_output_proj
        self.have_key_proj = have_key_proj
        self.P = nn.Parameter(torch.randn(n_head, d_out if not mlp and not have_output_proj else d, d) * winit / math.sqrt(d))
        self.Q = nn.Parameter(torch.randn(n_head, d, d) * winit / math.sqrt(d))
        if have_output_proj:
            self.output_proj = nn.Linear(n_head*d, d_out) if not mlp else nn.Linear(n_head*d, d)
        if have_key_proj:
            self.key_proj = nn.Parameter(torch.randn(n_head, d, d) * winit / math.sqrt(d))

        self.layer_norm = nn.LayerNorm(d_out if not mlp else d) if layer_norm else None
        self.mlp = nn.Sequential(
            nn.Linear(d, 2*d),
            nn.ReLU(),
            nn.Linear(2*d, d)
        ) if mlp else None
        self.projection = nn.Linear(d, d_out) if  mlp else None

    def attention(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Compute the softmax attention mechanism.

        Args:
            Z (torch.Tensor): Input tensor of shape (B, T, d).

        Returns:
            torch.Tensor: Output tensor after attention mechanism.
        """
        # Output: B x d x d
        B, T, d = Z.shape
        m, d_out, _ = self.P.shape

        # query
        Q = self.Q.reshape(m * d, d) # (md) x d
        QZ = Z @ Q.T # B x T x (md)
        QZ = QZ.reshape(B, T, m, d).transpose(1, 2) # B x m x T x d

        # value
        P = self.P.reshape(m * d_out, d) # (md) x d
        PZ = Z @ P.T # B x T x (md)
        PZ = PZ.reshape(B, T, m, d_out).transpose(1, 2) # B x m x T x d

        # key: normally we have another projection to produce multi head key
        if self.have_key_proj:
            key_proj = self.key_proj.reshape(m * d, d)
            KZ = Z @ key_proj.T
            KZ = KZ.reshape(B, T, m, d).transpose(1, 2)
            Z = KZ
        else:
            Z = Z.unsqueeze(1) # B x 1 x T x  d
            Z = Z.repeat(1, m, 1, 1) # B x m x T x d

        # attention
        # Create a causal mask to ensure that each position can only attend to previous positions
        QZ = (QZ @ Z.transpose(-2, -1)) / math.sqrt(self.d) # B x m x T x T
        causal_mask = torch.tril(torch.ones(T, T, device=Z.device)).unsqueeze(0).unsqueeze(0)  # 1 x 1 x T x T
        QZ = QZ.masked_fill(causal_mask == 0, float('-inf'))  # Apply the mask to the attention scores
        # softmax
        QZ = torch.softmax(QZ, dim=-1)
        # output
        output = torch.matmul(QZ, PZ) # B x m x T x d

        if self.have_output_proj:
            # we want to map  m x d part to d_out
            # B x T x md
            output = output.transpose(1, 2).reshape(B, T, m * d)
            output = self.output_proj(output)
            y = output
        else:
            y = output.mean(dim=1) # B x T x d

        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the LinearAttention module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying attention, layer norm, MLP, and projection.
        """
        residual = x

        x = self.attention(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x + residual)

        if self.mlp is not None:
            residual = x
            x = self.mlp(x)
            x = x + residual

        if self.projection is not None:
            x = self.projection(x)

        return x


class MultiLayerLinearAttention(nn.Module):
    """
    Multi-layer Linear Attention module that stacks multiple LinearAttention layers.
    """
    def __init__(self, d: int, d_out: int = 1, n_head: int = 1, n_layer: int = 1, mlp: bool = False, layer_norm: bool = False, softmax=False, have_output_proj=False, have_key_proj=False) -> None:
        """
        Initialize the MultiLayerLinearAttention module.

        Args:
            d (int): Input dimension.
            d_out (int): Output dimension.
            n_head (int): Number of attention heads.
            n_layer (int): Number of layers.
            mlp (bool): Whether to use MLP.
            layer_norm (bool): Whether to use Layer Normalization.
        """
        super(MultiLayerLinearAttention, self).__init__()
        self.d = d
        self.d_out = d_out
        self.n_head = n_head
        self.n_layer = n_layer
        self.softmax = softmax
        self.have_output_proj = have_output_proj
        self.have_key_proj = have_key_proj

        attn_module = functools.partial(SoftmaxAttention, have_output_proj=have_output_proj, have_key_proj=have_key_proj) if softmax else LinearAttention

        self.layers = nn.ModuleList(attn_module(d, d_out if l == n_layer-1 else d, n_head, mlp, layer_norm, 1.0) for l in range(n_layer))

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MultiLayerLinearAttention module.

        Args:
            Z (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through all layers.
        """
        for layer in self.layers:
            Z = layer(Z)
        return Z

def run_experiment(d: int = 3, T: int = 100, num_epochs: int = 500, N: int = 256, n_head: int = 2, n_layer: int = 1, lr: float = 0.01, batch_size: int = 64, d_out: Optional[int] = None, layer_norm: bool = False, mlp: bool = False, softmax: bool = False, output_proj: bool = False, key_proj: bool = False, seed: int = 0) -> List[float]:
    """
    Run an experiment to train the MultiLayerLinearAttention model.

    Args:
        d (int): Input dimension.
        T (int): Number of tokens.
        N (int): Number of samples.
        n_head (int): Number of attention heads.
        n_layer (int): Number of layers.
        lr (float): Learning rate.
        batch_size (int): Batch size.
        d_out (Optional[int]): Output dimension.
        layer_norm (bool): Whether to use Layer Normalization.
        mlp (bool): Whether to use MLP.

    Returns:
        List[float]: List of losses for each epoch.
    """
    torch.manual_seed(42)

    if d_out is None:
        d_out = d

    T = 100 #number of tokens

    # set the seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    true_P = torch.randn(1, d_out, d).to(DEVICE) / math.sqrt(d) #ground truth value matrix
    true_Q = torch.randn(1, d, d).to(DEVICE) / math.sqrt(d) #ground truth key-query matrix

    # Generate a random Z with i.i.d standard normal entries
    inputs = torch.randn(N, T, d).to(DEVICE) / math.sqrt(T)

    # Compute PZ(Z^TQZ) with true parameters
    targets = compute_expression(true_P, true_Q, inputs)

    # Validation inputs
    val_inputs = torch.randn(N, T, d).to(DEVICE) / math.sqrt(T)
    val_targets = compute_expression(true_P, true_Q, val_inputs)

    model = MultiLayerLinearAttention(d, d_out=d_out, n_head=n_head, n_layer=n_layer, layer_norm=layer_norm, mlp=mlp, softmax=softmax, have_output_proj=output_proj, have_key_proj=key_proj).to(DEVICE)
    # Define the optimizer
    # print(model)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    total_iterations = int(np.ceil(num_epochs * inputs.shape[0] / batch_size))
    # learning rate scheduler
    scheduler = TimmCosineLRScheduler(
        optimizer,
        t_initial=total_iterations,
        lr_min=lr * 0.01,
        warmup_t=total_iterations // 10,
        warmup_lr_init=1e-5,
        warmup_prefix=True
    )

    # Define the loss function (mean squared error)
    loss_fn = torch.nn.MSELoss(reduction='mean')

    @torch.no_grad()
    def evaluate(val_data):
        model.eval()
        val_loss = 0.0
        total = 0
        for i in range(0, val_data.shape[0], batch_size):
            input_batch = val_data[i:i+batch_size].to(DEVICE)
            target_batch = val_targets[i:i+batch_size].to(DEVICE)
            output = model.forward(input_batch)
            loss = loss_fn(output, target_batch)
            val_loss += loss.item()
            total += 1
        return val_loss / total


    # Training loop
    data = []
    total_iter = 0
    for epoch in tqdm(range(num_epochs)):
        epoch_loss = 0.0
        # Shuffle the dataset at the beginning of each epoch
        permutation = torch.randperm(inputs.shape[0])
        inputs = inputs[permutation]
        targets = targets[permutation]

        data.append(evaluate(val_inputs))
        # print(f"Epoch {epoch+1}, Val Loss: {data[-1]}")

        model.train()

        for i in range(0, inputs.shape[0], batch_size):
            # Get the mini-batch
            # Z : B x T x d
            input_batch = inputs[i:i+batch_size].to(DEVICE)
            # Y: B x T x 1
            target_batch = targets[i:i+batch_size].to(DEVICE)
            # Zero the gradients
            optimizer.zero_grad()


            output = model.forward(input_batch)
            loss = loss_fn(output, target_batch)

            # Backward pass
            loss.backward()

            # clip gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

            # Update parameters
            optimizer.step()

            total_iter += 1

            scheduler.step(total_iter)

            # Accumulate loss
            epoch_loss += loss.item()
        # print(f"Epoch {epoch+1}, Loss: {epoch_loss / (inputs.shape[0] / batch_size)}")
    return data


#Creates H matrix dependent on Z and the b coordinate of (a,b)
#arguments Z is data matrix
#argument b is the column
#feature does not change with a (the row of P_{a:})
#A new feature function for the folded regression problem that guarantees uniqueness
def fold_feature(Z, b):
    """
    Computes the vector v from the matrix Z as described.

    Parameters:
    Z (torch.Tensor): A tensor of shape (d, n)

    Returns:
    torch.Tensor: A tensor v of shape (d^3,)
    """
    Z = Z.T
    d, n = Z.shape
    #{j k} {l} distinct choices for j not equal to k is d choose 2 times d.
    #number of choices for j = k is d
    v = torch.zeros(int(d*d*(d-1)/2) + d**2, device=Z.device)
    index = 0
    for j in range(d):
        for k in range(j+1, d):
            for l in range(d):
                v[index] = torch.dot(Z[j], Z[k]) * Z[l, b]
                index += 1

    for j in range(d):
        for l in range(d):
            v[index] = torch.dot(Z[j], Z[j]) * Z[l, b]
            index += 1
    return v


#folds the parameters of P,Q into format given by fold_feature
def fold_params(P,Q):
    d = P.shape[0]
    par = int(d*d*(d-1)/2 + d**2)
    W = torch.zeros(par)
    index = 0
    for j in range(d):
        for k in range(j+1, d):
            for l in range(d):
                W[index] = P[a,j]*Q[k,l] + P[a,k]*Q[j,l]
                index += 1
    for j in range(d):
        for l in range(d):
            W[index] = P[a,j]*Q[j,l]
            index += 1
    return W


def train_with_poly_algorithm(batch_size=64, num_epochs=500, d=2, N=2048, T=100, lr=0.01, seed=0):
    # Initialize trainable parameters P and Q
    torch.manual_seed(seed)
    np.random.seed(seed)
    true_P = torch.randn(1, 1, d).to(DEVICE) / math.sqrt(d)  #ground truth value matrix
    true_Q = torch.randn(1, d, d).to(DEVICE) / math.sqrt(d)  #ground truth key-query matrix
    # Generate a random Z with i.i.d standard normal entries
    inputs = torch.randn(N, T, d).to(DEVICE) / math.sqrt(T)
    # Compute PZ(Z^TQZ) with true parameters
    targets = compute_expression(true_P, true_Q, inputs)

    # Validation inputs
    val_inputs = torch.randn(N, T, d).to(DEVICE) / math.sqrt(T)
    val_targets = compute_expression(true_P, true_Q, val_inputs)

    par = int(d*d*(d-1)/2 + d**2)
    W = (torch.randn(par) / math.sqrt(par))
    W = W.to(DEVICE)
    W.requires_grad = True
    # parameter
    # Define the optimizer
    optimizer = optim.Adam([W], lr=lr)

    # Define the loss function (mean squared error)
    loss_fn = torch.nn.MSELoss()

    @torch.no_grad()
    def evaluate(val_data):
        val_loss = 0.0
        total = 0
        for i in range(0, val_data.shape[0], batch_size):
            input_batch = val_data[i:i+batch_size].cpu()
            target_batch = val_targets[i:i+batch_size]
            input_data_converted = []
            for b in range(batch_size):
                time_data = [fold_feature(input_batch[b], t) for t in range(T)]
                input_data_converted.append(torch.stack(time_data))
            input_data_converted = torch.stack(input_data_converted)
            batch_covariances = input_data_converted.to(DEVICE, dtype=torch.float)
            batch_outputs = torch.matmul(batch_covariances, W).unsqueeze(-1)
            loss = loss_fn(batch_outputs, target_batch)
            val_loss += loss.item()
            total += 1
        return val_loss / total

    # Training loop
    poly_data = []

    for epoch in tqdm(range(num_epochs)):
        epoch_loss = 0.0
        num_batches = (N + batch_size - 1) // batch_size  # Calculate the number of batches
        poly_data.append(evaluate(val_inputs))
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, N)
            input_batch = inputs[start_idx:end_idx].cpu()
            # Get the batch data
            input_data_converted = []
            for b in range(batch_size):
                time_data = [fold_feature(input_batch[b], t) for t in range(T)]
                input_data_converted.append(torch.stack(time_data))
            input_data_converted = torch.stack(input_data_converted)
            batch_covariances = input_data_converted.to(DEVICE, dtype=torch.float)

            batch_results = targets[start_idx:end_idx]

            # Zero the gradients
            optimizer.zero_grad()
            # Forward pass for the batch
            batch_outputs = torch.matmul(batch_covariances, W).unsqueeze(-1)

            # Compute loss for the batch
            loss = loss_fn(batch_outputs, batch_results)

            # Backward pass
            loss.backward()

            # Update parameters
            optimizer.step()

            # Accumulate loss
            epoch_loss += loss.item()

    return poly_data

import fcntl
import pickle
import time

def safe_read_pickle(file_path):
    while True:
        try:
            with open(file_path, 'rb') as f:
                fcntl.flock(f, fcntl.LOCK_SH | fcntl.LOCK_NB)
                data = pickle.load(f)
                fcntl.flock(f, fcntl.LOCK_UN)
            return data
        except BlockingIOError:
            time.sleep(0.1)  # Wait for 100ms before retrying

def safe_write_pickle(data, file_path):
    while True:
        try:
            with open(file_path, 'wb') as f:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                pickle.dump(data, f)
                fcntl.flock(f, fcntl.LOCK_UN)
            return
        except BlockingIOError:
            time.sleep(0.1)  # Wait for 100ms before retrying

import argparse
import pandas as pd

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run hyperparameter experiments.")
    parser.add_argument("--seed", type=int, default=0, help="Seed value for the experiments")
    args = parser.parse_args()

    # Define the hyperparameter grid
    d_values = [2, 4]
    n_head_values = [1, 2, 4, 8]
    d_out_values = [1,]
    n_layer_values = [1, 2, 4]
    mlp_layer_norm_combinations = [(False, False), (True, True)]
    lr_values = [0.01, 0.001]
    batch_size_values = [32, 64]
    N_values = [512, 1024, 2048]  # Number of training points
    softmax_values = [True, False]
    output_proj_values = [True, False]
    key_proj_values = [True, False]
    seed_values = [args.seed]


    if not os.path.exists("poly_results.pkl"):
        poly_df = pd.DataFrame(columns=["seed", "d", "d_out", "lr", "batch_size", "N", "losses"])
        safe_write_pickle(poly_df, "poly_results.pkl")
    else:
        poly_df = safe_read_pickle("poly_results.pkl")

    for seed in seed_values:
        for N in N_values:
            for lr in lr_values:
                for batch_size in batch_size_values:
                    losses = train_with_poly_algorithm(N=N, seed=seed, lr=lr, batch_size=batch_size)
                    poly_df = safe_read_pickle("poly_results.pkl")
                    poly_df = pd.concat([poly_df, pd.DataFrame([{
                        "seed": seed, "d": 2, "d_out": 1, "lr": lr, "batch_size": batch_size, "N": N, "losses": losses
                    }])], ignore_index=True)
                    safe_write_pickle(poly_df, "poly_results.pkl")






    # File to save results
    results_file = "results_pickle.pkl"

    # Load existing results if the file exists
    if os.path.exists(results_file):
        df = safe_read_pickle(results_file)
    else:
        df = pd.DataFrame(columns=["seed", "d", "n_head", "d_out", "n_layer", "mlp", "layer_norm", "softmax", "output_proj", "key_proj", "lr", "batch_size", "N", "losses"])
        # Save empty dataframe
        safe_write_pickle(df, results_file)


    # Sweep over all hyperparameters using itertools
    from itertools import product

    hyperparameter_combinations = product(
        seed_values, d_values, n_head_values, d_out_values, n_layer_values,
        mlp_layer_norm_combinations, softmax_values, output_proj_values, key_proj_values, lr_values, batch_size_values, N_values
    )

    for setting in reversed(list(hyperparameter_combinations)):
        seed, d, n_head, d_out, n_layer, (mlp, layer_norm), softmax, output_proj, key_proj, lr, batch_size, N = setting
        if not softmax and (output_proj or key_proj):
            continue

        df = safe_read_pickle(results_file)

        if not ((df["d"] == d) &
                (df["n_head"] == n_head) &
                (df["d_out"] == d_out) &
                (df["n_layer"] == n_layer) &
                (df["mlp"] == mlp) &
                (df["layer_norm"] == layer_norm) &
                (df["lr"] == lr) &
                (df["batch_size"] == batch_size) &
                (df["N"] == N) &
                (df["softmax"] == softmax) &
                (df["output_proj"] == output_proj) &
                (df["seed"] == seed) &
                (df["key_proj"] == key_proj)).any():

            new_row = {
                "d": d, "n_head": n_head, "d_out": d_out, "n_layer": n_layer,
                "mlp": mlp, "layer_norm": layer_norm, "lr": lr,
                "batch_size": batch_size, "N": N,
                "softmax": softmax, "output_proj": output_proj, "key_proj": key_proj,
                "seed": seed,
            }

            losses = run_experiment(**new_row)

            new_row["losses"] = losses
            # read again
            df = safe_read_pickle(results_file)
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            safe_write_pickle(df, results_file)
        else:
            print(f"Skipping {setting} already completed")

    print("All experiments completed.")
