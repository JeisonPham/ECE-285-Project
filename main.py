import numpy as np
import torch
import sklearn.datasets
import struct
import torchvision

import sys
sys.path.insert(1, "D:\\PythonProjects\\ECE-285-Project\\utils")

import utils

from utils.preprocess import preprocess_data
from utils.loss import nll
from utils.helpers import generate_sign_patterns, get_out_string
from utils.cvxpy_model import cvxpy_solver
from utils.pytorch_model import sgd_solver
from utils.visualization import get_times_epoch_xaxis, plot_metrics_over_time, plot_all_models

from Noisy_MNIST import NoisyMNIST

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        # we can consider resizing if need be
        (0.1307,), (0.3081,))])
train_dataset = NoisyMNIST("data", train=True, download=True, transform=transform, target_transform=None, std=1)
test_dataset = NoisyMNIST("data", train=False, download=True, transform=transform, target_transform=None, std=1)

dummy_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=50000, shuffle=False,
    pin_memory=True, sampler=None)
for A, y in dummy_loader:
    pass

P, verbose = 50, True # SET verbose to True to see progress
sign_pattern_list, u_vector_list = generate_sign_patterns(A, P, verbose)
num_neurons = len(sign_pattern_list)
num_epochs, batch_size = 10, 100
beta_noncvx = 1e-3
learning_rate = 1e-3

beta_cvx = 2 * beta_noncvx
solver_type = "sgd" # pick: "sgd" or "LBFGS"
LBFGS_param = [10, 4] # these parameters are for the LBFGS solver
learning_rate = 1e-3
results_pt_relaxed = sgd_solver(train_dataset, test_dataset,
                                num_epochs, num_neurons, beta_cvx,
                                learning_rate, batch_size, solver_type,
                                LBFGS_param, D=50, rho=0, convex=True,
                                u_vector_list=u_vector_list, verbose=verbose,
                                eps=1e-2, last_n=10)