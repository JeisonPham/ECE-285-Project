import torch
from ConvexNN.models import ConvexReLU
from ConvexNN.utils import sample_gate_vectors


def generate_model(input_dim, num_neurons, output_dim, num_epochs, beta,
                   model_chain,learning_rate, batch_size, rho, device='cpu'):
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
    G = sample_gate_vectors(42, d=input_dim, num_neurons=num_neurons)
    model = ConvexReLU(G, c=output_dim, p=num_neurons, d=input_dim)
    train_dataset = NoisyMnist(..., model_chain) # should return Nwh x 9
    test_dataset = NoiseMnist(..., model_chain)