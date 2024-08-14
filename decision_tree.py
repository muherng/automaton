import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

mode = 'unitary'
#style = 'unique'
style = 'degenerate'


# Set random seed for reproducibility (optional)
torch.manual_seed(47)

# Create the fixed matrices P and Q with i.i.d standard normal entries

d = 4 #dimension of model d by d 
n = int(d/2) + 1 #number of tokens 
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

print('true_P: ', true_P)
print('true_Q: ', true_Q)

# Define the function to compute PZ(Z^TQZ)
def compute_expression(P, Q, Z):
    ZTQZ = torch.matmul(torch.matmul(Z.T, Q), Z)
    result = torch.matmul(P, torch.matmul(Z, ZTQZ))
    #print('QZ: ', torch.matmul(Q,Z))
    #print('PZ: ', torch.matmul(P,Z))
    #print('Z^TQZ: ', ZTQZ)
    return result

# Number of random choices of Z
num_samples = 2**12

# Initialize lists to store input Z's and results
Z_list = []
results = []

# Define the coordinate to fit (a, b)
a, b = int(d/2), n-1  # Example: coordinate (3,3)
print('a: ', a)
print('b: ', b)
def random_unitary_matrix(size):
    q, _ = torch.qr(torch.randn(size, size))
    return q
#generate synthetic outputs
#regenerating the unitary matrix for every sample 
#is critical for uniqueness of the regressor
#both unique and degnerate styles work
#when unitary matrix is fixed, the regressor is not unique
for _ in range(num_samples):
    zero_one = random_unitary_matrix(num_tokens)
    zero = zero_one[:,0].view(num_tokens,1)
    one = zero_one[:,1].view(num_tokens,1)
    zero_mat = torch.cat((zero, zero), dim=1) 
    one_mat = torch.cat((one, one), dim=1)

    #print('zero_one: ', zero_one)
    #print('zero mat: ', zero_mat)
    #print('one mat: ', one_mat)
    if mode == 'unitary':
        # Generate the top and bottom parts as random unitary matrices
        if style == 'unique': 
            top = zero_one
            prob = torch.tensor([0.5, 0.5])
            sample = torch.bernoulli(prob)
            #print('sample: ', sample)
            first = zero_one[:,int(sample[0])].view(num_tokens,1)
            second = zero_one[:,int(sample[1])].view(num_tokens,1)
            bottom = torch.cat((first,second), dim=1)  
            #print('bottom: ', bottom)
            #top = torch.randn((num_tokens,num_tokens))
            #bottom = torch.randn((num_tokens,num_tokens))
        if style == 'degenerate':
            top = zero_one
            # Define the probability tensor with probability 1/2
            prob = torch.tensor([0.5])
            # Draw a sample from the Bernoulli distribution
            sample = torch.bernoulli(prob)
            if sample == 1:
                bottom = one_mat
            else: 
                bottom = zero_mat 
            #bottom = one_mat
            #bottom = top
        # Generate the last column as zero or one with probability 1/2
        prob = torch.tensor([0.5])
        # Draw a sample from the Bernoulli distribution
        sample = torch.bernoulli(prob)
        if sample == 1:
            query = zero_one[:,1].view(num_tokens,1)
        else: 
            query = zero_one[:,0].view(num_tokens,1)
        
        #choose query to be standard normal
        query = torch.randn(num_tokens,1)

        shape = (num_tokens,1)
        # Fill the rest of the last column with zeros
        filler = torch.randn(shape)*1.0
        last_column = torch.cat((query,filler), dim=0).view(-1,1)

        # Concatenate the top and bottom parts with the last column
        Z = torch.cat((torch.cat((top, bottom), dim=0), last_column), dim=1)
    
    #print('Z: ', Z)
    
    # Compute PZ(Z^TQZ) with true parameters
    result = compute_expression(true_P, true_Q, Z)
    
    # Store the Z and the specific coordinate result
    Z_list.append(Z)
    #print('result: ', result)
    #print('result total: ', result[a:a+2,b])
    results.append(result[a, b])
    #raise ValueError('stop here')

# Convert lists to tensors
Z_tensor = torch.stack(Z_list)
results_tensor = torch.tensor(results)  
print('results_tensor: ', results_tensor)

#Creates H matrix dependent on Z and the b coordinate of (a,b) 
#arguments Z is data matrix
#argument b is the column 
#feature does not change with a (the row of P_{a:})
#A new feature function for the folded regression problem that guarantees uniqueness
def fold_feature(Z,b):
    """
    Computes the vector v from the matrix Z as described.

    Parameters:
    Z (torch.Tensor): A tensor of shape (d, n)

    Returns:
    torch.Tensor: A tensor v of shape (d^3,)
    """
    d, n = Z.shape
    #{j k} {l} distinct choices for j not equal to k is d choose 2 times d.  
    #number of choices for j = k is d 
    v = torch.zeros(int(d*d*(d-1)/2) + d**2)
    index = 0
    for j in range(d):
        for k in range(j+1, d):
            for l in range(d):
                v[index] = torch.inner(Z[j], Z[k])*Z[l,b]
                index += 1
    
    for j in range(d):
        for l in range(d):
            v[index] = torch.inner(Z[j], Z[j])*Z[l,b]
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


def compute_max_min_eigen(Z_tensor, b, num_samples):
    # Compute the vectors
    vectors = torch.stack([fold_feature(Z_tensor[i], b) for i in range(num_samples)])
    
    # Compute the covariance matrix using torch.cov
    covariance_matrix = torch.cov(vectors.T)

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

# Example usage
max_eigenvalue, min_eigenvalue, max_eigenvector, min_eigenvector = compute_max_min_eigen(Z_tensor, b, num_samples)
print("Max eigenvalue:", max_eigenvalue)
print("Min eigenvalue:", min_eigenvalue)
print("Max eigenvector:", max_eigenvector)
print("Min eigenvector:", min_eigenvector)

# Initialize trainable parameters P and Q
#d^3 for a'th entry of P, recall only regressing (a,b) coordinate
par = int(d*d*(d-1)/2 + d**2)
W = torch.randn(par, requires_grad=True)

# Define the optimizer
optimizer = optim.Adam([W], lr=0.01)

# Define the loss function (mean squared error)
loss_fn = torch.nn.MSELoss()

# Training loop
num_epochs = 80
poly_data = []
batch_size = 32  # Define the batch size

# Training loop
for epoch in range(num_epochs):
    epoch_loss = 0.0
    num_batches = (num_samples + batch_size - 1) // batch_size  # Calculate the number of batches

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_samples)

        # Get the batch data
        batch_covariances = torch.stack([fold_feature(Z_tensor[i], b) for i in range(start_idx, end_idx)])
        batch_results = results_tensor[start_idx:end_idx]

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass for the batch
        batch_outputs = torch.matmul(batch_covariances, W)

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
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')
        poly_data = poly_data + [epoch_loss]

target_regressor = fold_params(true_P,true_Q)
print('ground truth coefficients: ', target_regressor)
print('learned regressor: ', W)
l1error = torch.dist(target_regressor, W,p=1)
print('l1 error: ', l1error)

# Flatten the W tensor to get a 1D array of its entries
W_flat = W.flatten().detach().numpy()

# Create a bar plot
plt.figure(figsize=(10, 6))
plt.bar(range(len(W_flat)), W_flat)
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Bar Plot of Each Coefficient')
plt.show()

#check if inner product of target_regressor and fold_feature(Z,b)
#is equal to the target value 
def debug(target_regressor, Z, b):
    cov = fold_feature(Z,b)
    print('inner product: ', torch.inner(target_regressor, cov))
    #print('target value: ', results_tensor[0])

#loop over results_tensor and check if debug function works 
for i in range(1):
   Z = Z_tensor[i]
   cov = fold_feature(Z,b)
   print('iteration: ', i)
   print('cov: ', cov)
   print('learned label: ', torch.inner(W, cov))
   print('debug label: ', torch.inner(target_regressor, cov))
   print('true label: ', results_tensor[i])

#print('moe_data: ', moe_data)
#print('poly_data: ', poly_data)
#np.save('moe_data', np.array(moe_data))
#np.save('poly_data', np.array(poly_data))


