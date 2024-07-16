import torch
import torch.optim as optim

# Set random seed for reproducibility (optional)
torch.manual_seed(42)

# Create the fixed matrices P and Q with i.i.d standard normal entries
true_P = torch.randn(2, 2)
true_Q = torch.randn(2, 2)

# Define the function to compute PZ(Z^TQZ)
def compute_expression(P, Q, Z):
    ZTQZ = torch.matmul(torch.matmul(Z.T, Q), Z)
    result = torch.matmul(P, torch.matmul(Z, ZTQZ))
    return result

# Number of random choices of Z
num_samples = 100

# Initialize lists to store input Z's and results
Z_list = []
results = []

# Define the coordinate to fit (a, b)
a, b = 0, 5  # Example: top-left coordinate
d = 2
n = 30
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

# Initialize trainable parameters P and Q
P = torch.randn(2, 2, requires_grad=True)
Q = torch.randn(2, 2, requires_grad=True)

# Define the optimizer
optimizer = optim.Adam([P, Q], lr=0.01)

# Define the loss function (mean squared error)
loss_fn = torch.nn.MSELoss()

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for i in range(num_samples):
        Z = Z_tensor[i]
        target = results_tensor[i]
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = compute_expression(P, Q, Z)
        output_coordinate = output[a, b]
        
        # Compute loss
        loss = loss_fn(output_coordinate, target)
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        # Accumulate loss
        epoch_loss += loss.item()
    
    # Print average loss for each epoch
    if (epoch + 1) % 1 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / num_samples:.4f}')

# Print the learned parameters
print("Learned P:", P)
print("Learned Q:", Q)

import torch

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
Z = torch.randn(d, n)
v = feature(Z,b)
print(v)

# Initialize trainable parameters P and Q
W = torch.randn(d**3, requires_grad=True)

# Define the optimizer
optimizer = optim.Adam([W], lr=0.01)

# Define the loss function (mean squared error)
loss_fn = torch.nn.MSELoss()

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    epoch_loss = 0.0
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

