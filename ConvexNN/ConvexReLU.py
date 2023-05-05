import torch
import torch.nn as nn


class ConvexReLU(nn.Module):
    def __init__(self, d, num_neurons, num_classes=10):
        super().__init__()
        self.v = nn.Parameter(data=torch.zeros(num_neurons, d, num_classes), requires_grad=True)
        self.w = nn.Parameter(data=torch.zeros(num_neurons, d, num_classes), requires_grad=True)

    def forward(self, x, sign_patterns):
        sign_patterns = sign_patterns.unsqueeze(2)
        x = x.view(x.shape[0], -1)  # n x d

        Xv_w = torch.matmul(x, self.v - self.w)  # P x N x C

        # for some reason, the permutation is necessary. not sure why
        DXv_w = torch.mul(sign_patterns, Xv_w.permute(1, 0, 2))  # N x P x C
        y_pred = torch.sum(DXv_w, dim=1, keepdim=False)  # N x C

        return y_pred