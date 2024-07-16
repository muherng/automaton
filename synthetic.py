import torch
import torch.optim as optim

# Set random seed for reproducibility (optional)
torch.manual_seed(42)

# Create the fixed matrices P and Q with i.i.d standard normal entries

d = 3
n = 100

true_P = torch.randn(d, d)
true_Q = torch.randn(d, d)

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
a, b = 0, 1  # Example: top-left coordinate
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
P = torch.randn(d, d, requires_grad=True)
Q = torch.randn(d, d, requires_grad=True)

# Define the optimizer
optimizer = optim.Adam([P, Q], lr=0.01)

# Define the loss function (mean squared error)
loss_fn = torch.nn.MSELoss()

# Training loop
""" trans_data = []
num_epochs = 300
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
    trans_data = trans_data + [epoch_loss / num_samples] """

# Define batch size
batch_size = 32

# Training loop
trans_data = []
num_epochs = 300
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
            output = compute_expression(P, Q, Z)
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
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / (num_samples / batch_size):.4f}')
    trans_data = trans_data + [epoch_loss / num_samples]



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

print('trans_data: ', trans_data)
print('poly_data: ', poly_data)

import matplotlib.pyplot as plt

# Sample lists of different lengths
list1 = trans_data
list2 = poly_data

# Generate x values for each list
x1 = list(range(len(list1)))
x2 = list(range(len(list2)))

# Create the plot
plt.figure(figsize=(10, 6))

# Plot list1
plt.plot(x1, list1, label='Transformer Landscape', marker='o')

# Plot list2
plt.plot(x2, list2, label='Polynomial Relaxation Landscape', marker='x')

# Set y-axis limits
plt.ylim(0, 100)

# Add labels and title
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Learning One Layer Linear Attention: Adam LR 0.01')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()


