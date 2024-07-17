import matplotlib.pyplot as plt
import numpy as np

moe_data = np.load('moe_data.npy')
poly_data = np.load('poly_data.npy')

# Sample lists of different lengths
list1 = moe_data[0,:]
list2 = moe_data[1,:]
list3 = poly_data

# Generate x values for each list
x1 = list(range(len(list1)))
x2 = list(range(len(list2)))
x3 = list(range(len(list3)))

# Create the plot
plt.figure(figsize=(10, 6))

# Plot list1
plt.plot(x1, list1, label='Transformer Landscape', marker='o')

plt.plot(x2, list2, label='Mixture 2 Experts', marker='o')

# Plot list2
plt.plot(x3, list3, label='Polynomial Relaxation Landscape', marker='x')

# Set y-axis limits
plt.ylim(0, 1000)

# Add labels and title
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Learning One Layer Linear Attention: Adam LR=0.01 No Batching')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()