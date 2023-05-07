import torch
import torchvision
import numpy as np

__all__ = ["NoisyMNIST"]


class NoisyMNIST(torchvision.datasets.MNIST):
    """
    Derived class to build a noisy version of MNIST    
    
    """
    def __init__(self, root, train, download, transform, target_transform, 
                 kernel_size = None, stride = 1, mean=0.0, std=1):
        assert isinstance(kernel_size,int) or isinstance(kernel_size,None)
        assert isinstance(stride,int)
        assert stride >= 1
        # initializing the parent class
        super().__init__(root=root, train=train, download=download, 
                         transform=transform, target_transform=target_transform)
        # initializing the mean and standard deviation
        self.mean = mean
        self.std = std
        self.stride = stride
        self.kernel_size = kernel_size
    '''
    Deprecated after finding torch's unfold function
    def __convolution_patches__(self,tensor):
        # case where no convolutional patches are used
        if self.window_shape == None:
            return tensor
        kernel_x = self.window_shape[1]
        kernel_y = self.window_shape[2]
        if self.stride != 1:
            kernel_x += self.stride
            kernel_y += self.stride
        tensor = tensor.numpy()
        tensor = np.lib.stride_tricks.sliding_window_view(tensor,window_shape=(1,kernel_x,kernel_y))
        tensor = torch.tensor(tensor)
        tensor = torch.reshape(tensor,(1,tensor.shape[1]*tensor.shape[2],tensor.shape[4]*tensor.shape[5]))
        
        tensor = tensor[:,:,::self.stride]
        return tensor
    '''
    def __getitem__(self, ndx):
        baseImage, label = super().__getitem__(ndx)
        noisyImage = baseImage + torch.randn(baseImage.size()) * self.std + self.mean
        if self.kernel_size is not None:
            unfold = torch.nn.Unfold(kernel_size=self.kernel_size,stride=self.stride)
            noisyImage = unfold(noisyImage)
            noisyImage = torch.transpose(noisyImage,0,1)
        return noisyImage, baseImage.flatten()

transform = torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))])
train_dataset = NoisyMNIST("data", train=True, download=True, transform=transform, target_transform=None, std=0.25, kernel_size=3)
print(train_dataset[0][0].shape)
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
#class AddGaussianNoise(object):
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
#Convolutional Patches Sanity Check

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