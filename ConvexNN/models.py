import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import lab


def relu_solution_mapping(convex_model, remove_sparse: bool = False):
    v = convex_model.v.detach().numpy()
    w = convex_model.w.detach().numpy()

    weights = np.asarray([v, w])
    assert len(weights.shape) == 4

    weight_norms = (lab.sum(weights ** 2, axis=-1, keepdims=True)) ** (1 / 4)
    normalized_weights = lab.safe_divide(weights, weight_norms)

    num_classes = weights.shape[1]
    first_layer = None
    second_layer = []
    for c in range(num_classes):
        pre_zeros = [
            lab.zeros_like(weight_norms[0, c]) for i in range(2 * c)
        ]  # positive neurons
        post_zeros = [
            lab.zeros_like(weight_norms[0, c])
            for i in range(2 * (num_classes - c - 1))
        ]

        if first_layer is None:
            pre_weights = []
        else:
            pre_weights = [first_layer]

        first_layer = lab.concatenate(
            pre_weights
            + [
                normalized_weights[0][c],
                normalized_weights[1][c],
            ],
            axis=0,
        )

        w2 = lab.concatenate(
            pre_zeros
            + [
                weight_norms[0][c],
                -weight_norms[1][c],
            ]
            + post_zeros,
            axis=0,
        ).T
        second_layer.append(w2)

    second_layer = lab.concatenate(second_layer, axis=0)

    if remove_sparse:
        sparse_indices = lab.sum(first_layer, axis=1) != 0

        first_layer = first_layer[sparse_indices]
        second_layer = second_layer[:, sparse_indices]

    return first_layer, second_layer


class NonConvexReLU(nn.Module):
    def __init__(self, d, p, c):
        super().__init__()
        self.d = d
        self.p = p
        self.c = c

        self.W1 = nn.Parameter(data=torch.zeros(self.p, self.d), requires_grad=True)
        self.W2 = nn.Parameter(data=torch.zeros(self.c, self.p), requires_grad=True)

    def update_parameters(self, w1, w2):
        self.W1 = nn.Parameter(data=torch.from_numpy(w1), requires_grad=True)
        self.W2 = nn.Parameter(data=torch.from_numpy(w2), requires_grad=True)

    def forward(self, x):
        Z = x @ self.W1.T
        return F.relu(Z) @ self.W2.T


class ConvexReLU(nn.Module):
    def __init__(self, G, c, p, d):
        super().__init__()
        self.G = G
        if not torch.is_tensor(self.G):
            self.G = torch.from_numpy(self.G).float()
        self.d, self.p = G.shape
        self.c = c

        self.v = nn.Parameter(data=torch.zeros(c, p, d), requires_grad=True)
        self.w = nn.Parameter(data=torch.zeros(c, p, d), requires_grad=True)

    def __call__(self, x, D=None):
        if D is None:
            D = torch.ge(x @ self.G, 0).float()
        else:
            D = torch.from_numpy(D).float()

        p_dff = self.v - self.w
        return torch.einsum("ij, lkj, ik->il", x, p_dff, D)


if __name__ == "__main__":
    from utils import sample_gate_vectors, loss_func_cvxproblem


    def relu(x):
        return np.maximum(0, x)


    def drelu(x):
        return x >= 0

    N = 12
    d = 3
    num_neurons = 500
    c = 1
    beta = 1e-4

    G = sample_gate_vectors(42, d, num_neurons)
    print(G.shape)

    X = np.random.rand(N, d)
    dmat = np.empty((N, 0))

    ## Finite approximation of all possible sign patterns
    for i in range(int(1e2)):
        u = np.random.randn(d, 1)
        dmat = np.append(dmat, drelu(np.dot(X, u)), axis=1)

    dmat = (np.unique(dmat, axis=1))
    convexModel = ConvexReLU(G, c, dmat.shape[1], d)
    y = ((np.linalg.norm(X[:,0:d-1],axis=1)>1)-0.5)*2
    # y = [((np.linalg.norm(X[:,0:d-1],axis=1)>1)-0.5)*2, ((np.linalg.norm(X[:,0:d-1],axis=1)>1)-0.5)*3]
    # y = np.asarray(y).T

    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float()
    optimizer = torch.optim.SGD(convexModel.parameters(), lr=1e-4, momentum=0.9)

    convexModel.train()
    previous_loss = 1000
    while True:
        yhat = convexModel(X, dmat)
        optimizer.zero_grad()
        loss = loss_func_cvxproblem(yhat, y, convexModel, X, beta)
        if torch.abs(previous_loss - loss) < 1e-3:
            break
        previous_loss = loss
        loss.backward()
        optimizer.step()

    print(convexModel(X, dmat))
    first, second = relu_solution_mapping(convexModel)

    nonConvex = NonConvexReLU(d, num_neurons, c)
    nonConvex.update_parameters(first, second)
    print(nonConvex(X))


