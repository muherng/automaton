import torch
import torch.optim as optim
import math 
import numpy as np
import matplotlib.pyplot as plt
import pstats
import io
import cProfile

from certificate_helper import compute_expression, fold_feature, fold_params, compute_max_min_eigen, truePQ, training_loop_linear, training_loop_MHLA


def generate_mixture_data(d,prob,num_samples): 
    mode = 'mixed' # a mixture of random and unitary data
    style = 'unique'
    feature_mode = 'full'
    mixture_prob = prob
    # Number of random choices of Z
    #num_samples = 2**14
    # Set random seed for reproducibility (optional)
    #torch.manual_seed(47)

    # Create the fixed matrices P and Q with i.i.d standard normal entries

    #d = 8 #dimension of model d by d 
    n = int(d/2) + 1 #number of tokens 
    num_tokens = int(d/2) 

    if feature_mode == 'compressed':
        feature_length = int(d*d*(d-1)/2)
    elif feature_mode == 'full':
        feature_length = int(d*d*(d-1)/2 + d**2)
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

    #print('true_P: ', true_P)
    #print('true_Q: ', true_Q)

    # Define the coordinates to fit (a, b) to (d-1,b) which we write as range(a,d) x {b}
    a, b = int(d/2), n-1  
    k = d - a #dimension of prediction

    # Initialize lists to store input Z's and results
    Z_list = []
    results = torch.zeros((k,num_samples))
    def random_unitary_matrix(size):
        q, _ = torch.qr(torch.randn(size, size))
        return q
   
    for id in range(num_samples):
        if mode == 'random':
            # Generate the top and bottom parts as random unitary matrices
            top = torch.randn((num_tokens,num_tokens))
            bottom = torch.randn((num_tokens,num_tokens))
            
            # Normalize columns of top
            top_norms = torch.norm(top, dim=0, keepdim=True)
            top = top / top_norms

            # Normalize columns of bottom
            bottom_norms = torch.norm(bottom, dim=0, keepdim=True)
            bottom = bottom / bottom_norms
        if mode == 'unitary':
            # Generate the top and bottom parts as random unitary matrices
            top = random_unitary_matrix(num_tokens)
            bottom = random_unitary_matrix(num_tokens)
        if mode == 'mixed': 
            if torch.rand(1) < mixture_prob:
                top = random_unitary_matrix(num_tokens)
                bottom = random_unitary_matrix(num_tokens)
            else:
                top = torch.randn((num_tokens,num_tokens))
                bottom = torch.randn((num_tokens,num_tokens))
                # Normalize columns of top
                top_norms = torch.norm(top, dim=0, keepdim=True)
                top = top / top_norms

                # Normalize columns of bottom
                bottom_norms = torch.norm(bottom, dim=0, keepdim=True)
                bottom = bottom / bottom_norms
        
        if style == 'degenerate':
            bottom = top
        #pick a random column from top
        q = torch.randint(0,num_tokens,(1,))
        query = top[:,q].view(num_tokens,1)
        shape = (num_tokens,1)
        # Fill the rest of the last column with zeros
        filler = torch.randn(shape)*1.0
        last_column = torch.cat((query,filler), dim=0).view(-1,1)
        # Concatenate the top and bottom parts with the last column
        Z = torch.cat((torch.cat((top, bottom), dim=0), last_column), dim=1)
    
        # Compute PZ(Z^TQZ) with true parameters
        result = compute_expression(true_P, true_Q, Z)
        
        # Store the Z and the specific coordinate result
        Z_list.append(Z)
        results[:,id] = result[a:,b]
        id += 1

    # Convert lists to tensors
    Z_tensor = torch.stack(Z_list)
    results_tensor = results  

    # Define the arguments for fold_feature
    args = {'feature_mode': feature_mode, 'feature_length': feature_length,'b': b}    
    features = torch.stack([fold_feature(Z_tensor[i],**args) for i in range(num_samples)])

    torch.save({'Z_tensor': Z_tensor, 'features': features, 'results_tensor': results_tensor}, 'tensors.pth')
    return features,results_tensor

   
def train_on_data(d,heads,Z_tensor,features,num_samples,results_tensor,true_P,true_Q,feature_mode,feature_length): 
    a = int(d/2)
    n = int(d/2) + 1 #number of tokens 
    b = n-1 
    k = d-a #dimension of prediction

    # Example usage
    max_eigenvalue, min_eigenvalue, max_eigenvector, min_eigenvector = compute_max_min_eigen(features, num_samples)
    if min_eigenvalue < 0:
        print("Min eigenvalue:", min_eigenvalue)
        print("Warning: Min eigenvalue is negative")
        min_eigenvalue = 0

    #model_type = 'linear'
    model_type = 'MHLA'
    if model_type == 'linear':
        W = training_loop_linear(d,a,feature_length,num_samples,features,results_tensor)
    if model_type == 'MHLA':
        P_heads, Q_heads, loss = training_loop_MHLA(a,heads,num_samples,Z_tensor,results_tensor)
    args = {'feature_mode': feature_mode, 'feature_length': feature_length,'a': a,'b': b, 'k': k}
    target_regressor = fold_params(true_P,true_Q,**args)
    
    W = torch.zeros(target_regressor.shape)
    for head in range(heads):
        W += fold_params(P_heads[:,:,head],Q_heads[:,:,head],**args)
    W = W/heads
    #print('ground truth coefficients: ', target_regressor)
    #print('learned regressor: ', W)
    #print('W shape: ', W.shape)
    #l2error = torch.dist(target_regressor,W,p=2)/(W.shape[0]*W.shape[1])
    #print('target regressor shape: ', target_regressor.shape)
    l2error = torch.dist(target_regressor,W,p=2)
    #print('l2 error: ', l2error)

    # Output coordinates where target_regressor and W differ by more than 0.1
    # Flatten the W tensor to get a 1D array of its entries
    #W_flat = W.flatten().detach().numpy()

    # Create a bar plot
    #plt.figure(figsize=(10, 6))
    #plt.bar(range(len(W_flat)), W_flat)
    #plt.xlabel('Index')
    #plt.ylabel('Value')
    #plt.title('Bar Plot of Each Coefficient of Regressor For Fixed Transition')
    #plt.show()

    return min_eigenvalue, l2error, loss  

def run_experiment(heads,prob):
    d = 4
    num_samples = 2**14
    feature_mode = 'full'
    feature_length = int(d*d*(d-1)/2 + d**2)
    _,_ = generate_mixture_data(d,prob,num_samples)
    loaded_tensors = torch.load('tensors.pth')
    Z_tensor = loaded_tensors['Z_tensor']
    features = loaded_tensors['features']
    results_tensor = loaded_tensors['results_tensor']
    true_P, true_Q = truePQ(d)
    min_eigenvalue, l2error, loss = train_on_data(d,heads,Z_tensor,features,num_samples,
                                            results_tensor,true_P,
                                            true_Q,feature_mode,feature_length)
    print('min_eigenvalue: ', min_eigenvalue)
    print('l2error: ', l2error) 
    return min_eigenvalue, l2error, loss 


def main():
    start = 0.95
    step = 0.01
    end = 1.0 + step
    prob_list = torch.arange(start,end,step)
    heads_list = [2]
    index = -1
    # Convert lists to NumPy arrays
    eigen_array = np.zeros((len(heads_list),len(prob_list)))
    l2_array = np.zeros((len(heads_list),len(prob_list)))
    for heads in heads_list:
        index += 1
        eigen_list = []
        l2_list = [] 
        for i in range(len(prob_list)):
            min_eigenvalue, l2error, loss = run_experiment(heads,prob_list[i])
            print('heads: ', heads)
            print('prob: ', prob_list[i])
            print('eigen_list: ', eigen_list)
            print('l2_list: ', l2_list)
            if loss > 0.01:
                print('fail to train: ', loss)
                continue
            try : 
                epsilon = 1e-10  # Small value to avoid log(0) issues
                print('small number: ', min_eigenvalue + epsilon)
                eigen_list.append(-1*np.log(min_eigenvalue + epsilon))
                l2_list.append(l2error.item())
            except: 
                print('error')
                continue

        #print('eigen_array: ', eigen_array)
        #print('eigen_list: ', eigen_list)

        # Sort eigen_list and rearrange l2_list accordingly
        sorted_pairs = sorted(zip(eigen_list, l2_list))
        eigen_list, l2_list = zip(*sorted_pairs)

        # Convert back to lists
        eigen_list = list(eigen_list)
        l2_list = list(l2_list)

        eigen_array[index,:len(eigen_list)] = np.array(eigen_list)
        l2_array[index,:len(eigen_list)] = np.array(l2_list)
        
        # Save NumPy arrays to files
        np.save('eigen_array.npy', eigen_array)
        np.save('l2_array.npy', l2_array)

    # Plotting
    for i in range(eigen_array.shape[0]):
        plt.plot(eigen_array[i, :], l2_array[i, :], marker='o', label=f'Line {i+1}')

    plt.xlabel('Negative Log of Min Eigenvalue')
    plt.ylabel('L2 Error')
    plt.title('Negative Log of Min Eigenvalue vs L2 Error: d=4 Associative Memory')
    plt.grid(True)
    plt.legend()
    plt.show()


     
if __name__ == "__main__":
    #prob = 1.0
    #run_experiment(prob)
    main()

