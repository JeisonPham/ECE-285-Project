from ConvexNN.models import ConvexReluMLP
from ConvexNN.utils import sample_gate_vectors
import torch
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, latent_dim, num_classes=0):
        super(Encoder, self).__init__()
        self.num_classes = num_classes
        self.FC_input = nn.Linear(input_dim + num_classes, hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_mean = nn.Linear(hidden_dim, latent_dim)
        self.FC_var = nn.Linear(hidden_dim, latent_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)

        self.training = True

    def forward(self, x, label=None):
        x = x.view(x.shape[0], -1)
        if label is not None:
            label = torch.nn.functional.one_hot(label, num_classes=self.num_classes)
            x = torch.cat((x, label), dim=1)
        h_ = self.LeakyReLU(self.FC_input(x))
        h_ = self.LeakyReLU(self.FC_input2(h_))
        mean = self.FC_mean(h_)
        log_var = self.FC_var(h_)  # encoder produces mean and log of variance
        #             (i.e., parateters of simple tractable normal distribution "q"

        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, num_classes=0):
        super(Decoder, self).__init__()
        self.num_classes = num_classes
        self.FC_hidden = nn.Linear(latent_dim + num_classes, hidden_dim)
        # self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)

        self.LeakyReLU = nn.ReLU()

    def forward(self, x, label=None):
        x = x.view(x.shape[0], -1)
        if label is not None:
            label = torch.nn.functional.one_hot(label, num_classes=self.num_classes)
            x = torch.cat((x, label), dim=1)

        h = self.LeakyReLU(self.FC_hidden(x))
        # h = self.LeakyReLU(self.FC_hidden2(h))

        x_hat = self.FC_output(h)
        return x_hat


class Model(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(Model, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)  # sampling epsilon
        z = mean + var * epsilon  # reparameterization trick
        return z

    def forward(self, x, label=None):
        mean, log_var = self.Encoder(x, label)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))  # takes exponential function (log var -> var)
        x_hat = self.Decoder(z, label)

        return x_hat, mean, log_var