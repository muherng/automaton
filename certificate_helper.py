import torch
import torch.optim as optim
import numpy as np

#This file contains the helper functions for the certificate computation.   
# Define the function to compute PZ(Z^TQZ)
def compute_expression(P, Q, Z):
    ZTQZ = torch.matmul(torch.matmul(Z.T, Q), Z)
    result = torch.matmul(P, torch.matmul(Z, ZTQZ))
    #print('QZ: ', torch.matmul(Q,Z))
    #print('PZ: ', torch.matmul(P,Z))
    #print('Z^TQZ: ', ZTQZ)
    return result

def fold_feature(Z,**kwargs):
    """
    Parameters:
    Z (torch.Tensor): A tensor of shape (d, n)

    Returns:
    torch.Tensor: A tensor v of shape (d^3,)
    """
    for key, value in kwargs.items():
        if key == 'feature_mode':
            feature_mode = value
        elif key == 'feature_length':
            feature_length = value
        elif key == 'b':
            b = value

    d, n = Z.shape
    # {j k} {l} distinct choices for j not equal to k is d choose 2 times d.  
    # number of choices for j = k is d 
    v = torch.zeros(feature_length)
    
    # Vectorized computation for j != k
    idx = 0
    for j in range(d):
        for k in range(j+1, d):
            #inner_product = d**(-0.5)*torch.inner(Z[j], Z[k])
            inner_product = torch.inner(Z[j], Z[k])
            v[idx:idx+d] = inner_product * Z[:, b]
            idx += d
    
    if feature_mode == 'full':
        # Vectorized computation for j == j
        for j in range(d):
            #inner_product = d**(-0.5)*torch.inner(Z[j], Z[j])
            inner_product = torch.inner(Z[j], Z[j])
            v[idx:idx+d] = inner_product * Z[:, b]
            idx += d
    
    return v

 #folds the parameters of P,Q into format given by fold_feature
def fold_params(P,Q,**kwargs): 
    for key,value in kwargs.items():
        if key == 'feature_length':
            feature_length = value
        elif key == 'feature_mode':
            feature_mode = value
        elif key == 'a':
            a = value
        elif key == 'k':
            k = value
    d = P.shape[0]
    W = torch.zeros((k,feature_length))
    for row in range(k): 
        W[row,:] = fold_params_helper(P,Q,a+row,d,feature_length,feature_mode)
    return W

def fold_params_helper(P,Q,row,d,feature_length,feature_mode):
    index = 0
    W_row = torch.zeros(feature_length)
    for j in range(d):
        for k in range(j+1, d):
            for l in range(d):
                W_row[index] = P[row,j]*Q[k,l] + P[row,k]*Q[j,l]
                index += 1
    if feature_mode == 'full': 
        for j in range(d):
            for l in range(d):
                W_row[index] = P[row,j]*Q[j,l]
                index += 1
    return W_row

def compute_max_min_eigen(features, num_samples):
    
    # Compute the covariance matrix using torch.cov
    covariance_matrix = torch.cov(features.T)

    print('cov shape: ', covariance_matrix.shape)
    
    # Compute eigenvalues and eigenvectors using torch.linalg.eigh
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)
    print('eigenvalues: ', eigenvalues)
    
    # Find the maximum and minimum eigenvalues and their corresponding eigenvectors
    max_eigenvalue, max_index = torch.max(eigenvalues, 0)
    min_eigenvalue, min_index = torch.min(eigenvalues, 0)
    
    max_eigenvector = eigenvectors[:, max_index]
    min_eigenvector = eigenvectors[:, min_index]
    
    return max_eigenvalue.item(), min_eigenvalue.item(), max_eigenvector, min_eigenvector

#check if inner product of target_regressor and fold_feature(Z,b)
#is equal to the target value 
def debug(target_regressor, Z, b, results_tensor):
    cov = fold_feature(Z,b)
    print('inner product: ', torch.inner(target_regressor, cov))
    print('target value: ', results_tensor[0])

def truePQ(d):
    num_tokens = int(d/2) 
    # Create the identity and zero matrices once
    I = torch.eye(num_tokens)
    zero_matrix = torch.zeros(num_tokens, num_tokens)

    # Construct the 2d by 2d matrix
    top_row = torch.cat((zero_matrix, zero_matrix), dim=1)
    bottom_row = torch.cat((zero_matrix, I), dim=1)
    true_P = torch.cat((top_row, bottom_row), dim=0)

    # If you need to create another similar matrix, you can reuse I and zero_matrix
    top_row = torch.cat((I, zero_matrix), dim=1)
    bottom_row = torch.cat((zero_matrix, zero_matrix), dim=1)
    true_Q = torch.cat((top_row, bottom_row), dim=0)

    return true_P, true_Q

def training_loop_linear(d,a,feature_length,num_samples,features,results_tensor):
# Initialize trainable parameters P and Q
    #d^3 for a'th entry of P, recall only regressing (a,b) coordinate
    #W = torch.randn(feature_length, requires_grad=True)
    #k is dimension of prediction
    k = d - a
    W = torch.randn((k,feature_length), requires_grad=True)
    # Define the optimizer
    optimizer = optim.Adam([W], lr=0.01)

    # Define the loss function (mean squared error)
    loss_fn = torch.nn.MSELoss()

    # Training loop
    num_epochs = 100
    poly_data = []
    batch_size = 256  # Define the batch size

    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = (num_samples + batch_size - 1) // batch_size  # Calculate the number of batches

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)

            # Get the batch data
            #batch_covariances = torch.stack([fold_feature(Z_tensor[i], b) for i in range(start_idx, end_idx)])
            batch_covariances = torch.transpose(features[start_idx:end_idx,:], 0,1)
            batch_results = results_tensor[:,start_idx:end_idx]

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass for the batch
            batch_outputs = torch.matmul(W, batch_covariances)
            #raise ValueError('stop')
            # Compute loss for the batch
            loss = loss_fn(batch_outputs, batch_results)

            # Backward pass
            loss.backward()

            # Update parameters
            optimizer.step()

            # Accumulate loss
            epoch_loss += loss.item()

        # Print average loss for each epoch
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss/batch_size:.4f}')
            poly_data = poly_data + [epoch_loss]
    return W

#compute multi head linear attention
""" def MHLA(P_experts, Q_experts, Z):
    d, _, heads = P_experts.shape
    d,n = Z.shape
    sum_result = torch.zeros((d,n))  # Initialize the sum with a zero matrix of appropriate size
    
    for i in range(heads):
        P_i = P_experts[:, :, i]
        Q_i = Q_experts[:, :, i]
        sum_result += compute_expression(P_i, Q_i, Z)
    
    return sum_result/heads """

def MHLA(P_experts, Q_experts, batch_covariances):
    d, _, heads = P_experts.shape
    #print('shape: ', batch_covariances.shape)
    batch_size, d, n = batch_covariances.shape
    result = torch.zeros(batch_size, d, n, device=batch_covariances.device)
    
    for h in range(heads):
        P_h = P_experts[:, :, h]  # (d, d)
        Q_h = Q_experts[:, :, h]  # (d, d)
        
        for b in range(batch_size):
            Z = batch_covariances[b]  # (d, n)
            ZT = Z.T  # (n, d)
            
            ZTQZ = ZT @ Q_h @ Z  # (n, n)
            PZ = P_h @ Z  # (d, n)
            
            result[b] += PZ @ ZTQZ  # (d, n)
    
    return result / heads

def training_loop_MHLA(a,heads,num_samples,Z_tensor,results_tensor):
    # Initialize trainable parameters P experts and Q experts
    #k is dimension of prediction
    d,n = Z_tensor[0].shape
    k = d - a
    heads = 2
    P_experts = torch.randn(d,d,heads,requires_grad=True)
    Q_experts = torch.randn(d,d,heads,requires_grad=True)

    # Define the optimizer
    optimizer = optim.Adam([P_experts, Q_experts], lr=0.01)

    # Define the loss function (mean squared error)
    loss_fn = torch.nn.MSELoss()

    # Training loop
    num_epochs = 30
    poly_data = []
    batch_size = 256  # Define the batch size

    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = (num_samples + batch_size - 1) // batch_size  # Calculate the number of batches

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)

            # Get the batch data
            #batch_covariances = torch.stack([fold_feature(Z_tensor[i], b) for i in range(start_idx, end_idx)])

            #batch_covariances = torch.transpose(features[start_idx:end_idx,:], 0,1)
            batch_covariances = Z_tensor[start_idx:end_idx,:,:]
            batch_results = results_tensor[:,start_idx:end_idx]

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass for the batch
            #batch_outputs = torch.matmul(W, batch_covariances)
            batch_outputs = MHLA(P_experts, Q_experts, batch_covariances)
            batch_outputs = torch.transpose(batch_outputs[:,a:,n-1],0,1)
            #print('batch_outputs shape: ', batch_outputs.shape)
            #print('batch_results shape: ', batch_results.shape)
            #raise ValueError('stop')
            # Compute loss for the batch
            loss = loss_fn(batch_outputs, batch_results)

            # Backward pass
            loss.backward()

            # Update parameters
            optimizer.step()

            # Accumulate loss
            epoch_loss += loss.item()

        # Print average loss for each epoch
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss/batch_size:.4f}')
            poly_data = poly_data + [epoch_loss]
    return P_experts, Q_experts

