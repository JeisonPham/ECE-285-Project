import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


def sample_gate_vectors(
        seed: int,
        d: int,
        n_samples: int
) -> np.ndarray:

    rng = np.random.default_rng(seed)

    G = rng.standard_normal((d, n_samples))
    # G = sample_dense_gates(rng, d, n_samples)
    return G


def loss_func_cvxproblem(yhat, y, model, _x, beta):
    _x = _x.view(_x.shape[0], -1)

    # term 1
    loss = 0.5 * torch.norm(yhat - y) ** 2
    # term 2
    # for layer, p in enumerate(model.parameters()):
    #     if layer == 0:
    #         loss += beta / 2 * torch.norm(p) ** 2
        # else:
        #     loss += beta / 2 * sum([torch.norm(p[:, j], 1) ** 2 for j in range(p.shape[1])])

    return loss


def generate_sign_patterns(A, P, verbose=False):
    # generate sign patterns
    n, d = A.shape
    sign_pattern_list = []  # sign patterns
    u_vector_list = []  # random vectors used to generate the sign paterns
    umat = np.random.normal(0, 1, (d, P))
    sampled_sign_pattern_mat = (np.matmul(A, umat) >= 0)
    for i in range(P):
        sampled_sign_pattern = sampled_sign_pattern_mat[:, i]
        sign_pattern_list.append(sampled_sign_pattern)
        u_vector_list.append(umat[:, i])
    if verbose:
        print("Number of sign patterns generated: " + str(len(sign_pattern_list)))
    return len(sign_pattern_list), sign_pattern_list, u_vector_list


class UpdatedTestDataset(Dataset):
    def __init__(self, X, y):
        self.x = X
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        return self.x[item], self.y[item]


class PrepareData3D(Dataset):
    def __init__(self, X, y, z):
        if not torch.is_tensor(X):
            self.X = torch.from_numpy(X)
        else:
            self.X = X

        if not torch.is_tensor(y):
            self.y = torch.from_numpy(y)
        else:
            self.y = y

        if not torch.is_tensor(z):
            self.z = torch.from_numpy(z)
        else:
            self.z = z

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.z[idx]
