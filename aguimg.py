import torchvision.transforms as transforms
import torch
def color_jitter(x,p=0.8):
    C = x.shape[1]
    jitter = transforms.ColorJitter(0.8,0.8,0.8,0.2)
    for i in range(C):
        if p > torch.rand(1):
            x_band = x[:,i,:,:]
            x_band.unsqueeze_(1)
            x_band = jitter(x_band)
            x_band.squeeze_(1)
            x[:,i,:,:] = x_band
    return x

def augment(X):
    crop = transforms.RandomResizedCrop(19,interpolation=transforms.InterpolationMode.BICUBIC)
    flip = transforms.RandomHorizontalFlip(0.5)
    X = color_jitter(flip(crop(X)))
    # X2 = color_jitter(flip(crop(X)))
    return X