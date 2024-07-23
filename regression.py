import torch
import torch.optim as optim
import numpy as np

# Set random seed for reproducibility (optional)
torch.manual_seed(42)

# Create the fixed matrices P and Q with i.i.d standard normal entries

d = 2 #dimension of model d by d 
n = 100 #number of tokens 

true_P = torch.rand(d, d) #ground truth value matrix
true_Q = torch.randn(d, d) #ground truth key-query matrix

# Define the function to compute PZ(Z^TQZ)
def compute_expression(P, Q, Z):
    ZTQZ = torch.matmul(torch.matmul(Z.T, Q), Z)
    result = torch.matmul(P, torch.matmul(Z, ZTQZ))
    return result

# Number of random choices of Z
num_samples = 512

# Initialize lists to store input Z's and results
Z_list = []
results = []

# Define the coordinate to fit (a, b)
a, b = 0, 1  # Example: coordinate (0,1)

#generate synthetic outputs
for _ in range(num_samples):
    # Generate a random Z with i.i.d standard normal entries
    Z = 2 * torch.randint(0, 2, (d, n), dtype=torch.float) - 1
    print(Z)
    
    # Compute PZ(Z^TQZ) with true parameters
    result = compute_expression(true_P, true_Q, Z)
    
    # Store the Z and the specific coordinate result
    Z_list.append(Z)
    results.append(result[a, b])

# Convert lists to tensors
Z_tensor = torch.stack(Z_list)
results_tensor = torch.tensor(results)  
num_epochs = 1

#right now test up to two experts 
max_experts = 2
moe_data = np.zeros((max_experts,num_epochs))
for experts in range(1,max_experts+1): 
    P_experts = torch.randn(d,d,experts,requires_grad=True)
    Q_experts = torch.randn(d,d,experts,requires_grad=True)

    # Define the optimizer
    optimizer = optim.Adam([P_experts, Q_experts], lr=0.01)

    # Define the loss function (mean squared error)
    loss_fn = torch.nn.MSELoss()

    # Define batch size
    batch_size = 256

    # Training loop
    data = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        # Shuffle the dataset at the beginning of each epoch
        permutation = torch.randperm(num_samples)
        Z_tensor = Z_tensor[permutation]
        results_tensor = results_tensor[permutation]
        
        for i in range(0, num_samples, batch_size):
            # Get the mini-batch
            Z_batch = Z_tensor[i:i+batch_size]
            target_batch = results_tensor[i:i+batch_size]
            
            # Zero the gradients
            optimizer.zero_grad()
            
            batch_loss = 0.0
            for j in range(Z_batch.size(0)):
                Z = Z_batch[j]
                target = target_batch[j]
                
                # Forward pass
                for expert in range(experts):
                    P = P_experts[:,:,expert]
                    Q = Q_experts[:,:,expert] 
                    if expert == 0:
                        output = compute_expression(P, Q, Z)
                    else: 
                        output = output + compute_expression(P, Q, Z)
                output = output/experts
                output_coordinate = output[a, b]
                
                # Compute loss
                loss = loss_fn(output_coordinate, target)
                batch_loss += loss
            
            # Compute the average loss for the batch
            batch_loss = batch_loss / Z_batch.size(0)
            
            # Backward pass
            batch_loss.backward()
            
            # Update parameters
            optimizer.step()
            
            # Accumulate loss
            epoch_loss += batch_loss.item()
        
        # Print average loss for each epoch
        print(f'Experts: {experts} Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / (num_samples / batch_size):.4f}')
        data = data + [epoch_loss / (num_samples / batch_size)]
    moe_data[experts-1,:] = data

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
    v = torch.zeros(int(d*d*(d-1)/2 + d))
    index = 0
    for j in range(d):
        for k in range(j+1, d):
            for l in range(d):
                v[index] = torch.inner(Z[j], Z[k])*Z[l,b]
                index += 1
            
    for l in range(d):
        v[index] = n*Z[l,b]
        index += 1
    
    return v

#folds the parameters of P,Q into format given by fold_feature
def fold_params(P,Q): 
    d = P.shape[0]
    par = int(d*d*(d-1)/2 + d)
    W = torch.zeros(par)
    index = 0
    for j in range(d):
        for k in range(j+1, d):
            for l in range(d):
                W[index] = P[a,j]*Q[k,l] + P[a,k]*Q[j,l]
                index += 1
    for l in range(d):
        for j in range(d): 
            W[index] += P[a,j]*Q[j,l]
        index += 1
    return W


# Initialize trainable parameters P and Q
#d^3 for a'th entry of P, recall only regressing (a,b) coordinate
par = int(d*d*(d-1)/2 + d)
W = torch.randn(par, requires_grad=True)

# Define the optimizer
optimizer = optim.Adam([W], lr=0.01)

# Define the loss function (mean squared error)
loss_fn = torch.nn.MSELoss()

# Training loop
num_epochs = 100
poly_data = []
for epoch in range(num_epochs):
    epoch_loss = 0.0
    loss = 0
    for i in range(num_samples):
        Z = Z_tensor[i]
        cov = fold_feature(Z,b)
        target = results_tensor[i]
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = torch.inner(W,cov)
        
        # Compute loss
        loss = loss_fn(output, target)
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
    # Accumulate loss
    epoch_loss += loss.item()
    
    # Print average loss for each epoch
    if (epoch + 1) % 1 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / num_samples:.4f}')
        poly_data = poly_data + [epoch_loss / num_samples]

target_regressor = fold_params(true_P,true_Q)
print('target_regressor: ', target_regressor)
print('W: ', W)

#check if inner product of target_regressor and fold_feature(Z,b)
#is equal to the target value 
def debug(target_regressor, Z, b):
    cov = fold_feature(Z,b)
    print('inner product: ', torch.inner(target_regressor, cov))
    print('target value: ', results_tensor[0])

#loop over results_tensor and check if debug function works 
for i in range(1):
    Z = Z_tensor[i]
    cov = fold_feature(Z,b)
    print('cov: ', cov)
    print('inner product: ', torch.inner(target_regressor, cov))
    print('target value: ', results_tensor[i])

print('moe_data: ', moe_data)
print('poly_data: ', poly_data)
np.save('moe_data', np.array(moe_data))
np.save('poly_data', np.array(poly_data))

