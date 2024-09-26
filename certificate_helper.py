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
def fold_params(P,Q): 
    d = P.shape[0]
    W = torch.zeros((k,feature_length))
    for row in range(k): 
        W[row,:] = fold_params_helper(P,Q,a+row)
    return W

def fold_params_helper(P,Q,row):
    index = 0
    W_row = torch.zeros(feature_length)
    for j in range(d):
        for k in range(j+1, d):
            for l in range(d):
                W_row[index] = P[row,j]*Q[k,l] + P[row,k]*Q[j,l]
                #print('P[row,k]: ', P[row,k])
                #print('Q[j,l]: ', Q[j,l])
                #print('W_row[index]: ', W_row[index])
                index += 1
    if feature_mode == 'full': 
        for j in range(d):
            for l in range(d):
                W_row[index] = P[row,j]*Q[j,l]
                index += 1
    return W_row

