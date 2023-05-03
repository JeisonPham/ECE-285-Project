import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import torchvision
import numpy as np
from Noisy_MNIST import NoisyMNIST
import matplotlib.pyplot as plt

def visualize_dataset(x, y):
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(x[0])
    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(y[0])

    plt.show()


transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        # we can consider resizing if need be
        (0.1307,), (0.3081,))])
train_dataset = NoisyMNIST("data", train=True, download=True, transform=transform, target_transform=None)
x, y, _ = train_dataset[0]
visualize_dataset(x, y)