import torch
import torchvision
import numpy as np
import os
from torch.utils.data import Dataset

__all__ = ["NoisyMNIST", "ChainedNoisyMNIST", "create_dataset", "initialize_dataset"]


class NoisyMNIST(torchvision.datasets.MNIST):
    """
    Derived class to build a noisy version of MNIST    
    
    """

    def __init__(self, *args, mean=0.0, std=1, **kwargs):
        # initializing the parent class
        super().__init__(*args, **kwargs)
        # initializing the mean and standard deviation
        self.mean = mean
        self.std = std

    def __getitem__(self, ndx):
        baseImage, label = super().__getitem__(ndx)
        noisyImage = baseImage + torch.randn(baseImage.size()) * self.std + self.mean
        return noisyImage, baseImage.flatten()


class ChainedNoisyMNIST(Dataset):
    def __init__(self, folder_path):
        self.folder = folder_path
        self.labels = np.load(os.path.join(self.folder, "labels.npy"))
        self.images = np.load(os.path.join(self.folder, "noisy_images.npy"))
        self.gt_images = np.load(os.path.join(self.folder, "gt_images.npy"))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        gt_image = self.gt_images[item].reshape((1, 28, 28))
        noisy_image = self.images[item].reshape((1, 28, 28))
        label = self.labels[item]
        return torch.tensor(noisy_image), torch.tensor(gt_image.flatten()), torch.tensor(label)


def create_dataset(model, dataset, folder_path, device):
    labels = []
    images = []
    gt_images = []
    model.eval()

    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    for x, y, z in dataset:
        labels.append(z.detach().cpu().numpy())
        gt_images.append(y.detach().cpu().numpy())
        x = model(x.to(device))
        images.append(x.detach().cpu().numpy())
    images, gt_images, labels = np.concatenate(images, axis=0), np.concatenate(gt_images, axis=0), np.concatenate(
        labels, axis=0)
    np.save(os.path.join(folder_path, "labels.npy"), labels)
    np.save(os.path.join(folder_path, "noisy_images.npy"), images)
    np.save(os.path.join(folder_path, "gt_images.npy"), gt_images)


def initialize_dataset(std):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))])
    unfold = torch.nn.Unfold(kernel_size=3, stride=1)
    train_dataset = NoisyMNIST("data", train=True, download=True, transform=transform, std=std)

    temp_data = unfold(train_dataset[0][0]).T
    input_dim, _ = temp_data.shape
    train_indices = np.random.choice(len(train_dataset), 2000, replace=False)
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)

    test_dataset = NoisyMNIST("data", train=False, download=True, transform=transform, std=std)
    return train_dataset, test_dataset



""" Quick Sanity check test """

'''
import matplotlib.pyplot as plt
# loading in the data
transform = torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))])
Noisy_MNIST_FULL = NoisyMNIST("data", train=True, download=True, transform=transform, target_transform=None)
baseIm, noisyIm, label = Noisy_MNIST_FULL.__get_item__(5)
plt.subplot(1,2,1)
plt.imshow(baseIm.permute(1,2,0), cmap='gray')
plt.subplot(1,2,2)
plt.imshow(noisyIm.permute(1,2,0), cmap='gray')
plt.show()
'''
# Alternative option we might play around with...
# class to add gaussian white noise
# class AddGaussianNoise(object):
#    def __init__(self, mean=0., std=1.):
#        self.std = std
#        self.mean = mean

#    def __call__(self, tensor):
#        return tensor + torch.randn(tensor.size()) * self.std + self.mean

#    def __repr__(self):
#        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

# make sure that this Convolutional Patches is used after the ToTensor transformation in the Transforms composition
'''
class ConvolutionalPatches(object):
    def __init__(self, window_shape, stride = 1):
        assert isinstance(window_shape,tuple)
        assert isinstance(stride,int)
        assert stride >= 1
        # assumes that the object is already a tensor object, converts to numpy
        self.window_shape = window_shape
        self.stride = stride
    def __call__(self,tensor):
        tensor = tensor.numpy()
        tensor = np.lib.stride_tricks.sliding_window_view(tensor,window_shape=(1,self.window_shape[1]+self.stride,self.window_shape[2]+self.stride))
        tensor = torch.from_numpy(tensor)
        tensor = torch.reshape(tensor,(1,tensor.shape[1]*tensor.shape[2],tensor.shape[4]*tensor.shape[5]))
        tensor = tensor[:,:,::self.stride]
        return tensor
        

'''
# Convolutional Patches Sanity Check

'''


#Example of how to add ConvolutionalPatches class to the dataloader transforms in order to get the convolutional patches as rows in a matrix
import matplotlib.pyplot as plt
# loading in the baseline data
transform = torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))])
Noisy_MNIST_FULL = NoisyMNIST("data", train=True, download=True, transform=transform, target_transform=None)
baseIm, noisyIm = Noisy_MNIST_FULL.__getitem__(5)
print(baseIm.shape)

# loading in the Patched data
transform_Patches = torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,)),
                                ConvolutionalPatches((1,3,3))
                                 ])
                                
Noisy_MNIST_Patches = NoisyMNIST("data", train=True, download=True, transform=transform_Patches, target_transform=None)
baseIm_Patches, noisyIm_Patches, label = Noisy_MNIST_Patches.__getitem__(5)
print(baseIm_Patches.shape)

#print(noisyIm)

print(noisyIm_Patches)
'''
