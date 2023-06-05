import torch
from torch.utils.data import DataLoader
import torchvision
from ConvexNN.models import ConvexReLUCNN, ConvexReluMLP
from ConvexNN.utils import sample_gate_vectors
from ConvexNN.train import train
from Noisy_MNIST import NoisyMNIST, initialize_dataset
import numpy as np


def generate_model(num_neurons, output_dim, num_epochs, beta,
                   train_dataset, test_dataset,
                   model_chain, learning_rate, batch_size, rho, random_state=42, device='cpu'):
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
    unfold = torch.nn.Unfold(kernel_size=3, stride=1)
    print(train_dataset[0][0].shape)
    temp_data = unfold(train_dataset[0][0]).T

    input_dim, _ = temp_data.shape

    G = sample_gate_vectors(random_state, d=output_dim, n_samples=num_neurons).T
    # model = ConvexReLUCNN(G, output_dim, 9, unfold).to(device)
    model = ConvexReluMLP(G, output_dim=output_dim, input_dim=output_dim).to(device)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = train(model, train_dataloader, test_dataloader, num_epochs, learning_rate, beta, device='cuda')
    torch.save(model.state_dict(), "model.pt")
    return model, train_dataloader, test_dataloader


if __name__ == "__main__":
    train_dataset, test_dataset = initialize_dataset(1)
    model = generate_model(1000, 28 * 28, 100, 1e-4, train_dataset=train_dataset, test_dataset=test_dataset,
                   model_chain=None, learning_rate=1e-5,
                   batch_size=32, rho=1, random_state=42, device='cuda')
