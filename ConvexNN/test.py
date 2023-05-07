from scnn.optimize import optimize
from scnn.regularizers import NeuronGL1
import torchvision
import torch
from Noisy_MNIST import NoisyMNIST

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        # we can consider resizing if need be
        (0.1307,), (0.3081,))])
train_dataset = NoisyMNIST("data", train=True, download=True, transform=transform, target_transform=None, std=1)
test_dataset = NoisyMNIST("data", train=False, download=True, transform=transform, target_transform=None, std=1)

# data extraction
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=1000, shuffle=True,
    pin_memory=True, sampler=None)

for X_train, y_train in train_loader:
    pass

X_train = X_train.detach().numpy()
y_train = y_train.detach().numpy()

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=1000, shuffle=False,
    pin_memory=True)

for X_test, y_test in test_loader:
    pass

X_test = X_test.detach().numpy()
y_test = y_test.detach().numpy()

model, metrics = optimize(formulation="relu",
                          max_neurons=500,
                          X_train=X_train,
                          y_train=y_train,
                          regularizer=NeuronGL1(0.001),
                          verbose=True,
                          device="cuda")

model(X_train)