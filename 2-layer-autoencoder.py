import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from scnn.private.utils.data import gen_classification_data

from scnn.optimize import optimize
from scnn.regularizers import NeuronGL1
from Noisy_MNIST import NoisyMNIST


# loading in the Noisy MNIST Dataset
transform = torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                # we can consider resizing if need be
                                 (0.1307,), (0.3081,))])
Noisy_MNIST_FULL_train = NoisyMNIST("data", train=True, download=True, transform=transform, target_transform=None)
train_set_len = len(Noisy_MNIST_FULL_train)
print("Size of train set: ", train_set_len)
Noisy_MNIST_FULL_test = NoisyMNIST("data",train=False,download=True,transform=transform,target_transform=None)
test_set_len = len(Noisy_MNIST_FULL_test)
print("Size of test set: ", test_set_len)

y_example, x_example, label_example = Noisy_MNIST_FULL_test.__get_item__(5)
print(y_example.shape)
dim = torch.flatten(y_example).shape[0]

# initializing the dimensions of the dataset that will be used to train in giant tensors
X_train = torch.zeros(train_set_len,dim)
y_train = torch.zeros(train_set_len,dim)
X_test =  torch.zeros(test_set_len,dim)
y_test =  torch.zeros(test_set_len,dim)

print("Training Tensor Dimensions: ", X_train.shape)
print("Testing Tensor Dimensions: ", X_test.shape)

# converting the training data
for ndx in range(len(Noisy_MNIST_FULL_train)):
    baseIm, noisyIm, label = Noisy_MNIST_FULL_train.__get_item__(ndx)
    #print(baseIm.shape)
    X_train[ndx,:] = torch.flatten(noisyIm)
    y_train[ndx,:] = torch.flatten(baseIm)

# converting the test data
for ndx in range(len(Noisy_MNIST_FULL_test)):
    baseIm, noisyIm, label = Noisy_MNIST_FULL_test.__get_item__(ndx)
    X_test[ndx,:] = torch.flatten(noisyIm)
    y_test[ndx,:] = torch.flatten(baseIm)

# building the convex network
max_neurons = 250

# regularization term
lam = 0.001

# right now this requires too much RAM

'''
cvx_model, metrics = optimize("relu", 
                          max_neurons,
                          X_train, 
                          y_train, 
                          X_test, 
                          y_test, 
                          regularizer=NeuronGL1(lam),
                          verbose=True,  
                          device="cpu")

print(f"Hidden Layer Size: {cvx_model.parameters[0].shape[0]}")

'''

perm = torch.randperm(X_train.size(0))
idx = perm[:15000]
X_train_reduced = X_train[idx,:]
y_train_reduced = y_train[idx,:]
print(X_train_reduced.shape)

#perm = torch.randperm(X_test.size(0))
#idx = perm[:2500]
#X_test_reduced = X_test[idx,:]
#y_test_reduced = y_test[idx,:]
#print(X_test_reduced.shape)

'''
cvx_model_reduced, metrics_reduced = optimize("relu", 
                          max_neurons,
                          X_train_reduced, 
                          y_train_reduced, 
                          X_test, 
                          y_test, 
                          regularizer=NeuronGL1(lam),
                          verbose=True,  
                          device="cpu")

print(f"Hidden Layer Size: {cvx_model_reduced.parameters[0].shape[0]}")
'''

X_train_reduced_downsampled = X_train_reduced[:,0:X_train.shape[1]:2]
y_train_reduced_downsampled = y_train_reduced[:,0:X_train.shape[1]:2]

X_test_downsampled = X_test[:,0:X_train.shape[1]:2]
y_test_downsampled = y_test[:,0:X_train.shape[1]:2]
# removing everything unnecessary to training to free up memory
del Noisy_MNIST_FULL_test
del Noisy_MNIST_FULL_train
del X_test
del y_test
del X_train
del y_train
del X_train_reduced
del y_train_reduced
#print(X_train_reduced_downsampled.shape)

cvx_model_reduced_downsampled, metrics_reduced_downsampled = optimize("relu", 
                          max_neurons,
                          X_train_reduced_downsampled, 
                          y_train_reduced_downsampled, 
                          X_test_downsampled, 
                          y_test_downsampled, 
                          regularizer=NeuronGL1(lam),
                          verbose=True,  
                          device="cpu")

