import torch
import torch.optim as optim
import numpy as np

# Set random seed for reproducibility (optional)
torch.manual_seed(42)

# Create the fixed matrices P and Q with i.i.d standard normal entries

d = 3 #dimension of model d by d 
n = 100 #number of tokens 

true_P = torch.randn(d, d) #ground truth value matrix
true_Q = torch.randn(d, d) #ground truth key-query matrix

# Define the function to compute PZ(Z^TQZ)
def compute_expression(P, Q, Z):
    ZTQZ = torch.matmul(torch.matmul(Z.T, Q), Z)
    result = torch.matmul(P, torch.matmul(Z, ZTQZ))
    return result

# Number of random choices of Z
num_samples = 256

# Initialize lists to store input Z's and results
Z_list = []
results = []

# Define the coordinate to fit (a, b)
a, b = 0, 1  # Example: coordinate (0,1)

#generate synthetic outputs
for _ in range(num_samples):
    # Generate a random Z with i.i.d standard normal entries
    Z = torch.randn(d, n)
    
    # Compute PZ(Z^TQZ) with true parameters
    result = compute_expression(true_P, true_Q, Z)
    
    # Store the Z and the specific coordinate result
    Z_list.append(Z)
    results.append(result[a, b])

# Convert lists to tensors
Z_tensor = torch.stack(Z_list)
results_tensor = torch.tensor(results)  
num_epochs = 1000

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
def feature(Z,b):
    """
    Computes the vector v from the matrix Z as described.

    Parameters:
    Z (torch.Tensor): A tensor of shape (d, n)

    Returns:
    torch.Tensor: A tensor v of shape (d^3,)
    """
    d, n = Z.shape
    v = torch.zeros(d**3)

    for j in range(d):
        for k in range(d):
            for l in range(d):
                v[j * d * d + k * d + l] = torch.inner(Z[j], Z[k]) * Z[l,b]
    
    return v

# Example usage
#Z = torch.randn(d, n)
#v = feature(Z,b)
#print(v)

# Initialize trainable parameters P and Q
#d^3 for a'th entry of P, recall only regressing (a,b) coordinate
W = torch.randn(d**3, requires_grad=True)

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
        cov = feature(Z,b)
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

print('moe_data: ', moe_data)
print('poly_data: ', poly_data)
np.save('moe_data', np.array(moe_data))
np.save('poly_data', np.array(poly_data))

