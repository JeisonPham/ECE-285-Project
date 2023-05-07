import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import dill
import pickle
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import time
import scipy
from scipy.sparse.linalg import LinearOperator
import torch
import sklearn.linear_model
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.nn as nn
import argparse
import random
import torchvision
from Noisy_MNIST import NoisyMNIST
import torch.nn.functional as F


def visualize_dataset(x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(x, cmap='gray')
    ax = fig.add_subplot(1, 3, 2)
    ax.imshow(y, cmap='gray')
    ax = fig.add_subplot(1, 3, 3)
    ax.imshow(z, cmap='gray')
    plt.show()


transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        # we can consider resizing if need be
        (0.1307,), (0.3081,))])
train_dataset = NoisyMNIST("data", train=True, download=True, transform=transform, target_transform=None, std=2)
test_dataset = NoisyMNIST("data", train=False, download=True, transform=transform, target_transform=None, std=2)


class FCNetwork(nn.Module):
    def __init__(self, H, num_classes=10, input_dim=3072):
        self.num_classes = num_classes
        super(FCNetwork, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(input_dim, H, bias=False), nn.ReLU())
        self.layer2 = nn.Linear(H, num_classes, bias=False)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        out = self.layer2(self.layer1(x))
        return out


def loss_func_primal(yhat, y, model, beta):
    loss = 0.5 * torch.norm(yhat - y) ** 2

    # l2 norm on first layer weights, l1 squared norm on second layer
    for layer, p in enumerate(model.parameters()):
        if layer == 0:
            loss += beta / 2 * torch.norm(p) ** 2
        else:
            loss += beta / 2 * sum([torch.norm(p[:, j], 1) ** 2 for j in range(p.shape[1])])

    return loss


def validation_primal(model, testloader, beta, device):
    test_loss = 0
    test_correct = 0
    model.eval()
    for ix, (_x, _y) in enumerate(testloader):
        _x = Variable(_x).float().to(device)
        _y = Variable(_y).float().to(device)

        yhat = model(_x).float()

        loss = loss_func_primal(yhat, _y, model, beta)

        test_loss += loss.item()

    x = _x[0].detach().cpu().numpy().reshape(28, 28)
    y = _y[0].detach().cpu().numpy().reshape(28, 28)
    z = yhat[0].detach().cpu().numpy().reshape(28, 28)
    visualize_dataset(x, y, z)
    return test_loss


def sgd_solver_pytorch_v2(ds, ds_test, num_epochs, num_neurons, beta,
                          learning_rate, batch_size, solver_type, schedule,
                          LBFGS_param, verbose=False,
                          num_classes=10, D_in=3 * 1024, test_len=10000,
                          train_len=50000, device='cuda'):
    device = torch.device(device)
    # D_in is input dimension, H is hidden dimension, D_out is output dimension.
    H, D_out = num_neurons, num_classes
    # create the model
    model = FCNetwork(H, D_out, D_in).to(device)

    if solver_type == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif solver_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # ,
    elif solver_type == "adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)  # ,
    elif solver_type == "adadelta":
        optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)  # ,
    elif solver_type == "LBFGS":
        optimizer = torch.optim.LBFGS(model.parameters(), history_size=LBFGS_param[0], max_iter=LBFGS_param[1])  # ,

    # arrays for saving the loss and accuracy
    losses = np.zeros((int(num_epochs * np.ceil(train_len / batch_size))))
    accs = np.zeros(losses.shape)
    losses_test = np.zeros((num_epochs + 1))
    accs_test = np.zeros((num_epochs + 1))
    times = np.zeros((losses.shape[0] + 1))
    times[0] = time.time()

    losses_test[0] = validation_primal(model, ds_test, beta, device)  # loss on the entire test set

    if schedule == 1:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               verbose=verbose,
                                                               factor=0.5,
                                                               eps=1e-12)
    elif schedule == 2:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)

    iter_no = 0
    for i in range(num_epochs):
        model.train()
        for ix, (_x, _y) in enumerate(ds):
            # =========make input differentiable=======================
            _x = Variable(_x).to(device)
            _y = Variable(_y).to(device)

            # ========forward pass=====================================
            yhat = model(_x).float()

            loss = loss_func_primal(yhat, _y, model, beta)

            optimizer.zero_grad()  # zero the gradients on each pass before the update
            loss.backward()  # backpropagate the loss through the model
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()  # update the gradients w.r.t the loss

            losses[iter_no] += loss.item()  # loss on the minibatch

        iter_no += 1
        times[iter_no] = time.time()

        # get test loss and accuracy
        losses_test[i + 1] = validation_primal(model, ds_test, beta,
                                                                 device)  # loss on the entire test set

        if i % 1 == 0:
            print("Epoch [{}/{}], loss: {} acc: {}, test loss: {} test acc: {}".format(i, num_epochs,
                                                                                       np.round(losses[iter_no - 1], 3),
                                                                                       np.round(accs[iter_no - 1], 3),
                                                                                       np.round(losses_test[i + 1],
                                                                                                3) / test_len, np.round(
                    accs_test[i + 1] / test_len, 3)))
        if schedule > 0:
            scheduler.step(losses[iter_no - 1])

    return losses, accs, losses_test / test_len, accs_test / test_len, times, model


beta = 1e-3 # regularization parameter
num_epochs1, batch_size =  100, 1000 #
num_neurons = 4096 # number of neurons is equal to number of hyperplane arrangements

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    pin_memory=True, sampler=None)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=1000, shuffle=False,
    pin_memory=True)

print('Extracting the data')
dummy_loader= torch.utils.data.DataLoader(
    train_dataset, batch_size=50000, shuffle=False,
    pin_memory=True, sampler=None)
for A, y,  in dummy_loader:
    pass
Apatch=A.detach().clone()

A = A.view(A.shape[0], -1)
n,d=A.size()

solver_type = "sgd"  # pick: "sgd", "adam", "adagrad", "adadelta", "LBFGS"
schedule = 0  # learning rate schedule (0: Nothing, 1: ReduceLROnPlateau, 2: ExponentialLR)
LBFGS_param = [10, 4]  # these parameters are for the LBFGS solver
learning_rate = 1e-6

## SGD1 constant
print('SGD1-training-mu={}'.format(learning_rate))
results_noncvx_sgd1 = sgd_solver_pytorch_v2(train_loader, test_loader, num_epochs1, num_neurons, beta,
                                            learning_rate, batch_size, solver_type, schedule,
                                            LBFGS_param, verbose=True,
                                            num_classes=28 * 28, D_in=d, train_len=n)