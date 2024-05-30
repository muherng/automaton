import numpy as np
import matplotlib.pyplot as plt

    
data_dict = {}
data = []
models = ['enc_hybrid','enc_transformer','enc-dec_transformer']
for model in models:
    for states in range(2,20):
        for word_length in range(1,20): 
            file = 'saved_models/' + f'{model}s{states}w{word_length}_acc.npy'
            data_dict[(model,states,word_length)] = np.load(file)

data = {}
for model in models:
    for states in range(2,20):
        data_row = []
        data_sum = 0
        for word_length in range(1,20): 
            data_sum = data_sum + data_dict[(model,states,word_length)].size
            if data[(model,states)]: 
                data[(model,states)] = data[(model,states)].append(data_sum)
            else: 
                data[(model,states)] = data_sum
                
for states in range(2,20):             
    # Plotting the array
    plt.figure(figsize=(10, 5))  # Set the figure size
    for model in models: 
        plt.plot(data[(model,states)], marker='o', label=model)  # Plot first data set with circle markers
    plt.title(f"Data Required for 97% Validation Accuracy: Number of States {i+2}")  # Title of the plot
    plt.xlabel("Word Length")  # Label for the x-axis
    plt.ylabel("Data Requirement")  # Label for the y-axis
    #plt.ylim(0, 1)  # Set the y-axis to go from 0 to 1
    plt.grid(True)  # Show grid lines for better visibility
    plt.legend()  # Show legend to identify each data set
    # Save the plot as a PNG file
    plt.savefig(f'plots/{states}.png', format='png', dpi=300)  # Save as PNG with 300 dpi
    plt.show()  # Display the plot as usual

