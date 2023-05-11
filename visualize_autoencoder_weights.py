import torch
import matplotlib.pyplot as plt
import numpy as np
# loading in the model
model = torch.load("model.pt",map_location=torch.device('cpu'))
print(model)
print(model.keys())
# 
hist_count = len(model.keys())
layer_names = ["v","w"]
counter = 1
for key in layer_names:
    print(model[key].shape)
    cur_layer_tensor = model[key]
    for kernel in range(cur_layer_tensor.shape[1]):
        cur_kernel_weights = cur_layer_tensor[:,kernel,:]
        cur_kernel_weights = torch.flatten(cur_kernel_weights)
        cur_kernel_weights = cur_kernel_weights.numpy()
        counts, bins = np.histogram(cur_kernel_weights, bins = 256)
        plt.subplot(6,3,counter)
        plt.hist(bins[:-1], bins, weights=counts)
        plt.title("Layer: " + str(key) + " Kernel: " + str(kernel), fontsize=10, x=.85, y=.75)
        counter += 1

plt.suptitle("Kernel Weight Histograms")
#plt.tight_layout()
#plt.subplots(layout="constrained")
plt.show()
    # creating the histogram variables
    
    # plotting the histogram
    #plt.hist(bins[:-1], bins, weights=counts)
#print(model["v"].shape)