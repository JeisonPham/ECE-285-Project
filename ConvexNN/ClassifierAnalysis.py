import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import validation_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import torchvision
from Noisy_MNIST import NoisyMNIST

transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = NoisyMNIST("data", train=True, download=True, transform=transform, std=1)
test_dataset = NoisyMNIST("data", train=False, download=True, transform=transform, std=1)

train_images = []
train_labels = []
for _, image, label in train_dataset:
    train_images.append(image.detach().numpy())
    train_labels.append(label)

test_images = []
test_labels = []
for _, image, label in test_dataset:
    test_images.append(image.detach().numpy())
    test_labels.append(label)