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
    def __init__(self, G, output_dim, input_dim):
        super().__init__()
        self.G = G
        if not torch.is_tensor(self.G):
            self.G = torch.from_numpy(self.G).float()

        num_neurons, _ = self.G.shape

        self.v = nn.Parameter(data=torch.zeros(num_neurons, input_dim, output_dim), requires_grad=True)
        self.w = nn.Parameter(data=torch.zeros(num_neurons, input_dim, output_dim), requires_grad=True)
        self.G = nn.Parameter(data=self.G, requires_grad=False)

    def __call__(self, x):
        temp_x = torch.einsum("mn, ink->imnk", self.G, x)
        p_diff = self.v - self.w
        return torch.einsum("ijkl, jlm->im", temp_x, p_diff)


if __name__ == "__main__":
    from utils import sample_gate_vectors, loss_func_cvxproblem
    import torch
    from torch.nn import Unfold

    import torchvision
    from Noisy_MNIST import NoisyMNIST

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            # we can consider resizing if need be
            (0.1307,), (0.3081,))])
    train_dataset = NoisyMNIST("data", train=True, download=True, transform=transform,
                               target_transform=None, std=0.25, unfold_settings=dict(kernel_size=3, stride=1))

    print(train_dataset[0][0].shape)
