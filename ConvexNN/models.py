import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ConvexNN.utils import sample_gate_vectors


# import lab
#
#
# def relu_solution_mapping(convex_model, remove_sparse: bool = False):
#     v = convex_model.v.detach().numpy()
#     w = convex_model.w.detach().numpy()
#
#     weights = np.asarray([v, w])
#     assert len(weights.shape) == 4
#
#     weight_norms = (lab.sum(weights ** 2, axis=-1, keepdims=True)) ** (1 / 4)
#     normalized_weights = lab.safe_divide(weights, weight_norms)
#
#     num_classes = weights.shape[1]
#     first_layer = None
#     second_layer = []
#     for c in range(num_classes):
#         pre_zeros = [
#             lab.zeros_like(weight_norms[0, c]) for i in range(2 * c)
#         ]  # positive neurons
#         post_zeros = [
#             lab.zeros_like(weight_norms[0, c])
#             for i in range(2 * (num_classes - c - 1))
#         ]
#
#         if first_layer is None:
#             pre_weights = []
#         else:
#             pre_weights = [first_layer]
#
#         first_layer = lab.concatenate(
#             pre_weights
#             + [
#                 normalized_weights[0][c],
#                 normalized_weights[1][c],
#             ],
#             axis=0,
#         )
#
#         w2 = lab.concatenate(
#             pre_zeros
#             + [
#                 weight_norms[0][c],
#                 -weight_norms[1][c],
#             ]
#             + post_zeros,
#             axis=0,
#         ).T
#         second_layer.append(w2)
#
#     second_layer = lab.concatenate(second_layer, axis=0)
#
#     if remove_sparse:
#         sparse_indices = lab.sum(first_layer, axis=1) != 0
#
#         first_layer = first_layer[sparse_indices]
#         second_layer = second_layer[:, sparse_indices]
#
#     return first_layer, second_layer


class NonConvexReLU(nn.Module):
    def __init__(self, d, p, c, num_classes=0):
        super().__init__()
        self.d = d
        self.p = p
        self.c = c
        self.num_classes = num_classes

        self.W1 = nn.Parameter(data=torch.zeros(self.p, self.d + num_classes), requires_grad=True)
        self.W2 = nn.Parameter(data=torch.zeros(self.c, self.p), requires_grad=True)

    def update_parameters(self, w1, w2):
        self.W1 = nn.Parameter(data=torch.from_numpy(w1), requires_grad=True)
        self.W2 = nn.Parameter(data=torch.from_numpy(w2), requires_grad=True)

    def forward(self, x, label=None):
        x = x.view(x.shape[0], -1)
        # if label is not None:
        #     label = torch.nn.functional.one_hot(label, num_classes=self.num_classes)
        #     x = torch.cat((x, label), dim=1)

        Z = x @ self.W1.T
        return F.relu(Z) @ self.W2.T


class ConvexReluMLP(torch.nn.Module):
    def __init__(self, d, num_neurons, output_dims, num_classes=10):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ConvexReluMLP, self).__init__()

        self.num_classes = num_classes
        # P x d x C
        self.v = torch.nn.Parameter(data=torch.zeros(num_neurons, d + num_classes, output_dims), requires_grad=True)
        self.w = torch.nn.Parameter(data=torch.zeros(num_neurons, d + num_classes, output_dims), requires_grad=True)
        self.G = torch.nn.Parameter(data=torch.from_numpy(sample_gate_vectors(42, d + num_classes, num_neurons).astype(np.float32)), requires_grad=False)

    def forward(self, x, label):
        # sign_patterns = torch.from_numpy(sample_gate_vectors(42, x.shape[0], num_neurons)).to(x.get_device())
        x = x.view(x.shape[0], -1)
        if label is not None:
            label = torch.nn.functional.one_hot(label, num_classes=self.num_classes)
            x = torch.cat((x, label), dim=1)
        sign_patterns = (x @ self.G >= 0)
        sign_patterns = sign_patterns.unsqueeze(2)
        x = x.view(x.shape[0], -1)  # n x d

        Xv_w = torch.matmul(x, self.v - self.w)  # P x N x C

        # for some reason, the permutation is necessary. not sure why
        DXv_w = torch.mul(sign_patterns, Xv_w.permute(1, 0, 2))  # N x P x C
        y_pred = torch.sum(DXv_w, dim=1, keepdim=False)  # N x C

        return y_pred


class ConvexReLUCNN(nn.Module):
    def __init__(self, G, output_dim, input_dim, unfold):
        super().__init__()
        self.G = G
        if not torch.is_tensor(self.G):
            self.G = torch.from_numpy(self.G).float()

        num_neurons, _ = self.G.shape

        self.v = nn.Parameter(data=torch.zeros(num_neurons, input_dim, output_dim), requires_grad=True)
        self.w = nn.Parameter(data=torch.zeros(num_neurons, input_dim, output_dim), requires_grad=True)
        self.G = nn.Parameter(data=self.G, requires_grad=False)
        self.unfold = unfold

    def __call__(self, x):
        x = self.unfold(x)
        x = (torch.einsum("mn, ikn->imnk", self.G, x))
        p_diff = self.v - self.w
        return torch.einsum("ijkl, jlm->im", x, p_diff)


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
