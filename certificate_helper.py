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
