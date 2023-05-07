import torch
from torch.utils.data import DataLoader
import torchvision
from ConvexNN.models import ConvexReLU
from ConvexNN.utils import sample_gate_vectors
from ConvexNN.train import train
from Noisy_MNIST import NoisyMNIST
import numpy as np


def generate_model(num_neurons, output_dim, num_epochs, beta,
                   model_chain, learning_rate, batch_size, rho, device='cpu'):
    """

    :param input_dim: The input dimension of the model. For CNN convex formulation it should be kernel size
    :param num_neurons: The number of neurons, can be adjusted through evolutionary methods
    :param output_dim: the number of features that should be predicted
    :param num_epochs:
    :param beta:
    :param model_chain: Should be a list of already trained models
    :param learning_rate:
    :param batch_size:
    :param rho:
    :param device:
    :return:
    """

    device = torch.device(device)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = NoisyMNIST("data", train=True, download=True, transform=transform,
                               std=1, unfold_settings=dict(kernel_size=3, stride=1))
    input_dim, _ = train_dataset[0][0].shape
    train_indices = np.random.choice(len(train_dataset), 1000, replace=False)
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)

    test_dataset = NoisyMNIST("data", train=False, download=True, transform=transform,
                              std=1, unfold_settings=dict(kernel_size=3, stride=1))

    test_indices = np.random.choice(len(test_dataset), 1000, replace=True)
    test_dataset = torch.utils.data.Subset(test_dataset, test_indices)

    G = sample_gate_vectors(42, d=input_dim, n_samples=num_neurons).T
    model = ConvexReLU(G, output_dim, 9).to(device)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, shuffle=False)
    test_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    train(model, train_dataloader, test_dataloader, num_epochs, learning_rate, beta, device='cuda')


if __name__ == "__main__":
    generate_model(1, 28 * 28, 100, 1e-4, None, learning_rate=1e-5,
                   batch_size=25, rho=1, device='cuda')
