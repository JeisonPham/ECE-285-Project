import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import time
from .utils import *
from .ConvexReLU import ConvexReLU

def visualize_dataset(x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(x, cmap='gray')
    ax = fig.add_subplot(1, 3, 2)
    ax.imshow(y, cmap='gray')
    ax = fig.add_subplot(1, 3, 3)
    ax.imshow(z, cmap='gray')
    plt.show()


def validation_cvxproblem(model, testloader, u_vectors, beta, rho, device, print=False):
    test_loss = 0
    test_correct = 0
    test_noncvx_cost = 0

    with torch.no_grad():
        for ix, (_x, _y) in enumerate(testloader):
            _x = Variable(_x).to(device)
            _y = Variable(_y).to(device)
            _x = _x.view(_x.shape[0], -1)
            _z = (torch.matmul(_x, torch.from_numpy(u_vectors).float().to(device)) >= 0)

            yhat = model(_x, _z).float()

            loss = loss_func_cvxproblem(yhat, _y, model, _x, _z, beta, rho, device)

            test_loss += loss.item()

    if print:
        x = _x[0].detach().cpu().numpy().reshape(28, 28)
        y = _y[0].detach().cpu().numpy().reshape(28, 28)
        z = yhat[0].detach().cpu().numpy().reshape(28, 28)
        visualize_dataset(x, y, z)

    return test_loss


def sgd_solver_cvxproblem(ds, ds_test, num_epochs, num_neurons, beta,
                          learning_rate, batch_size, rho, u_vectors,
                          solver_type, LBFGS_param, verbose=False, other_models=[],
                          n=60000, d=28 * 28, num_classes=28 * 28, device='cpu'):
    device = torch.device(device)

    # create the model
    model = ConvexReLU(d, num_neurons, num_classes).to(device)

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
    losses = np.zeros((int(num_epochs * np.ceil(n / batch_size))))
    accs = np.zeros(losses.shape)
    noncvx_losses = np.zeros(losses.shape)

    losses_test = np.zeros((num_epochs + 1))
    accs_test = np.zeros((num_epochs + 1))
    noncvx_losses_test = np.zeros((num_epochs + 1))

    times = np.zeros((losses.shape[0] + 1))
    times[0] = time.time()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           verbose=verbose,
                                                           factor=0.5,
                                                           eps=1e-12)

    model.eval()
    losses_test[0] = validation_cvxproblem(model, ds_test, u_vectors, beta, rho,device, True)  # loss on the entire test set

    iter_no = 0
    print('starting training')
    for i in range(num_epochs):
        model.train()
        for ix, (_x, _y, _z) in enumerate(ds):
            # =========make input differentiable=======================

            _x = Variable(_x).to(device)
            _y = Variable(_y).to(device)
            _z = Variable(_z).to(device)

            for m in other_models:
                _x = m(_x, _z).float()

            # ========forward pass=====================================
            yhat = model(_x, _z).float()

            loss = loss_func_cvxproblem(yhat, _y, model, _x, _z, beta, rho, device) / len(_y)
            # =======backward pass=====================================
            optimizer.zero_grad()  # zero the gradients on each pass before the update
            loss.backward()  # backpropagate the loss through the model
            optimizer.step()  # update the gradients w.r.t the loss

            losses[iter_no] += loss.item()  # loss on the minibatch

        iter_no += 1
        times[iter_no] = time.time()

        model.eval()
        # get test loss and accuracy
        losses_test[i + 1] = validation_cvxproblem(model, ds_test,
                                                                                                u_vectors, beta, rho,
                                                                                                device)  # loss on the entire test set

        if i % 1 == 0:
            print(
                "Epoch [{}/{}], TRAIN: noncvx/cvx loss: {}, {} acc: {}. TEST: noncvx/cvx loss: {}, {} acc: {}".format(i,
                                                                                                                      num_epochs,
                                                                                                                      np.round(
                                                                                                                          noncvx_losses[
                                                                                                                              iter_no - 1],
                                                                                                                          3),
                                                                                                                      np.round(
                                                                                                                          losses[
                                                                                                                              iter_no - 1],
                                                                                                                          3),
                                                                                                                      np.round(
                                                                                                                          accs[
                                                                                                                              iter_no - 1],
                                                                                                                          3),
                                                                                                                      np.round(
                                                                                                                          noncvx_losses_test[
                                                                                                                              i + 1],
                                                                                                                          3) / 10000,
                                                                                                                      np.round(
                                                                                                                          losses_test[
                                                                                                                              i + 1],
                                                                                                                          3) / 10000,
                                                                                                                      np.round(
                                                                                                                          accs_test[
                                                                                                                              i + 1] / 10000,
                                                                                                                          3)))

        scheduler.step(losses[iter_no - 1])

    for param in model.parameters():
        param.requires_grad = False

    validation_cvxproblem(model, ds_test, u_vectors, beta, rho, device, True)
    return model
