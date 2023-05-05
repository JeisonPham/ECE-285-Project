import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision
import torch
from torch.autograd import Variable
import numpy as np
from torch.utils.data import DataLoader
from ConvexNN.utils import PrepareData3D, generate_sign_patterns
from ConvexNN.train import sgd_solver_cvxproblem
from Noisy_MNIST import NoisyMNIST

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        # we can consider resizing if need be
        (0.1307,), (0.3081,))])
train_dataset = NoisyMNIST("data", train=True, download=True, transform=transform, target_transform=None, std=1)
test_dataset = NoisyMNIST("data", train=False, download=True, transform=transform, target_transform=None, std=1)

# data extraction
print('Extracting the data')
dummy_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=50000, shuffle=False,
    pin_memory=True, sampler=None)
for A, y in dummy_loader:
    pass
Apatch = A.detach().clone()

A = A.view(A.shape[0], -1)
n, d = A.size()

# problem parameters
P, verbose = 512, True
beta = 1e-3  # regularization parameter
num_epochs1, batch_size = 100, 1000  #
num_neurons = P  # number of neurons is equal to number of hyperplane arrangements

# create dataloaders
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    pin_memory=True, sampler=None)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=1000, shuffle=False,
    pin_memory=True)

rho = 1e-2  # coefficient to penalize the violated constraints
solver_type = "adam"  # pick: "sgd", "adam", "adagrad", "adadelta", "LBFGS"
LBFGS_param = [10, 4]
batch_size = 1000
num_epochs2, batch_size = 1, 1000

#  Convex
print('Generating sign patterns')
num_neurons, sign_pattern_list, u_vector_list = generate_sign_patterns(A, P, verbose)
sign_patterns = np.array([sign_pattern_list[i].int().data.numpy() for i in range(num_neurons)])
u_vectors = np.asarray(u_vector_list).reshape((num_neurons, A.shape[1])).T

ds_train = PrepareData3D(X=A, y=y, z=sign_patterns.T)
ds_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=1000, shuffle=False,
    pin_memory=True)

#  Convex1
learning_rate = 1e-6  # 1e-6 for sgd
print('Convex Random1-mu={}'.format(learning_rate))
model1 = sgd_solver_cvxproblem(ds_train, test_loader, num_epochs2, num_neurons, beta,
                                     learning_rate, batch_size, rho, u_vectors, solver_type, LBFGS_param,
                                     verbose=True,
                                     n=n, device='cuda')

device = torch.device('cuda')
new_a = []
for ix, (_x, _y, _z) in enumerate(ds_train):
    # =========make input differentiable=======================

    _x = Variable(_x).to(device)
    _z = Variable(_z).to(device)
    new_a.append(model1(_x, _z).detach().cpu().numpy())

new_test = []
for ix, (_x, _y) in enumerate(test_loader):
    # =========make input differentiable=======================

    _x = Variable(_x).to(device)
    _z = (torch.matmul(_x, torch.from_numpy(u_vectors).float().to(device)) >= 0)
    new_test.append(model1(_x, _z).detach().cpu().numpy())


A = np.concatenate(new_a, axis=0)
A = torch.from_numpy(A)
num_neurons, sign_pattern_list, u_vector_list = generate_sign_patterns(A, P, verbose)
sign_patterns = np.array([sign_pattern_list[i].int().data.numpy() for i in range(num_neurons)])
u_vectors = np.asarray(u_vector_list).reshape((num_neurons, A.shape[1])).T

ds_train = PrepareData3D(X=A, y=y, z=sign_patterns.T)
ds_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)

test_dataset = torch.from_numpy(np.concatenate(new_test, axis=0))
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=1000, shuffle=False,
    pin_memory=True)

model2 = sgd_solver_cvxproblem(ds_train, test_loader, num_epochs2, num_neurons, beta,
                                     learning_rate, batch_size, rho, u_vectors, solver_type, LBFGS_param,other_models=[model1],
                                     verbose=True,
                                     n=n, device='cuda')