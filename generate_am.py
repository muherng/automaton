import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pstats
import io
import cProfile

from certificate_helper import compute_expression, fold_feature, fold_params, compute_max_min_eigen, truePQ

def generate_mixture_data(d,prob,num_samples): 
    mode = 'mixed' # a mixture of random and unitary data
    style = 'unique'
    feature_mode = 'full'
    mixture_prob = prob
    # Number of random choices of Z
    #num_samples = 2**14
    # Set random seed for reproducibility (optional)
    torch.manual_seed(47)

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

    torch.save({'features': features, 'results_tensor': results_tensor}, 'tensors.pth')
    return features,results_tensor

   
def train_on_data(d,features,num_samples,results_tensor,true_P,true_Q,feature_mode,feature_length): 
    a = int(d/2)
    n = int(d/2) + 1 #number of tokens 
    b = n-1

    # Example usage
    max_eigenvalue, min_eigenvalue, max_eigenvector, min_eigenvector = compute_max_min_eigen(features, num_samples)
    print("Min eigenvalue:", min_eigenvalue)

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
    num_epochs = 300
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
        if (epoch + 1) % 1 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss/batch_size:.4f}')
            poly_data = poly_data + [epoch_loss]

    args = {'feature_mode': feature_mode, 'feature_length': feature_length,'a': a,'b': b, 'k': k}
    target_regressor = fold_params(true_P,true_Q,**args)
    #print('ground truth coefficients: ', target_regressor)
    #print('learned regressor: ', W)
    #print('W shape: ', W.shape)
    #l2error = torch.dist(target_regressor,W,p=2)/(W.shape[0]*W.shape[1])
    #print('target regressor shape: ', target_regressor.shape)
    l2error = torch.dist(target_regressor,W,p=2)
    print('l2 error: ', l2error)

    # Output coordinates where target_regressor and W differ by more than 0.1
    # Flatten the W tensor to get a 1D array of its entries
    W_flat = W.flatten().detach().numpy()

    # Create a bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(W_flat)), W_flat)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Bar Plot of Each Coefficient of Regressor For Fixed Transition')
    plt.show()

    return min_eigenvalue, l2error  

def main():
    d = 8
    prob = 0.9
    num_samples = 2**14
    feature_mode = 'full'
    feature_length = int(d*d*(d-1)/2 + d**2)
    _,_ = generate_mixture_data(d,prob,num_samples)
    loaded_tensors = torch.load('tensors.pth')
    features = loaded_tensors['features']
    results_tensor = loaded_tensors['results_tensor']
    true_P, true_Q = truePQ(d)
    min_eigenvalue, l2error = train_on_data(d,features,num_samples,results_tensor,true_P,true_Q,feature_mode,feature_length)
    print('min_eigenvalue: ', min_eigenvalue)
    print('l2error: ', l2error)

if __name__ == "__main__":
    main()

