import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import time
from ConvexNN.utils import *
from ConvexNN.models import ConvexReLU


def visualize_dataset(x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(x, cmap='gray')
    ax = fig.add_subplot(1, 3, 2)
    ax.imshow(y, cmap='gray')
    ax = fig.add_subplot(1, 3, 3)
    ax.imshow(z, cmap='gray')
    plt.show()


def validation_cvxproblem(model, testloader, beta, device, print=False):
    test_loss = 0
    test_correct = 0
    test_noncvx_cost = 0

    with torch.no_grad():
        for ix, (_x, _y) in enumerate(testloader):
            _x = Variable(_x).to(device)
            _y = Variable(_y).to(device)

            yhat = model(_x).float()

            loss = loss_func_cvxproblem(yhat, _y, model, _x, beta, device)

            test_loss += loss.item()

    if print:
        x = _x[0].detach().cpu().numpy().reshape(28, 28)
        y = _y[0].detach().cpu().numpy().reshape(28, 28)
        z = yhat[0].detach().cpu().numpy().reshape(28, 28)
        visualize_dataset(x, y, z)

    return test_loss


def train_one_epoch(model, optimizer, dataloader, beta, device):
    model.train()
    losses = 0
    for index, (x, y) in enumerate(dataloader):
        x = Variable(x).to(device)
        y = Variable(y).to(device)

        optimizer.zero_grad()
        yhat = model(x)

        loss = loss_func_cvxproblem(yhat, y, model, x, beta)
        loss.backward()

        optimizer.step()

        losses += loss.item()
    return losses

def train(model, train_dataloader, test_dataloader, epochs, learning_rate, beta, device='cpu'):
    device = torch.device(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, factor=0.5, eps=1e-12)
    for epoch in range(epochs):
        model.train()
        loss = train_one_epoch(model, optimizer, train_dataloader, beta, device)

if __name__ == "__main__":
    D = np.ones((18, 8)) * 2
    X = np.random.rand(1000, 8, 2)
    u = np.random.rand(18, 2, 7)

    print(np.einsum("mn, ink->imnk", D, X).shape)
    N = []
    for i in range(X.shape[0]):
        M = []
        for m in range(D.shape[0]):
            temp = D[m, :].reshape(-1, 1) * X[i, :, :]
            M.append(temp)
        N.append(M)
    print(np.sum(np.asarray(N) == np.einsum("mn, ink->imnk", D, X)) / (1000 * 18 * 8 * 2))

    intermidiate = np.einsum("mn, ink->imnk", D, X)
    print(np.einsum("ijkl, jlm->im", intermidiate, u).shape)

    N = []
    for i in range(intermidiate.shape[0]):
        pass

    # values = []
    # for i in range(X.shape[0]):
    #     x1 = X[i, ...]
    #     temp = 0
    #     for j in range(u.shape[0]):
    #         u1 = u[j, ...]
    #         temp += np.sum(x1 @ u1, axis=0)
    #     values.append(temp)
    #
    # print(np.sum(np.einsum("ijk, lkm ->im", X, u) - np.asarray(values)))