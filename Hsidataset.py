import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


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

class HsiDataset_tra(Dataset):
    def __init__(self, data, label):
        self.data = data.reshape(-1, 19, 19, 30)
        self.label = label
        # self.transform = transform
        self.classes = label.max() + 1

    def __getitem__(self, i):
        # img = self.data[i]
        # tensor_trans = transforms.ToTensor()
        # tn = transforms.Normalize(np.zeros(30), np.ones(30))
        # img = tensor_trans(img)
        # img = tn(img)
        # targets = self.label[i]
        img1 = self.data[i, :, :, :15]
        img2 = self.data[i, :, :, 15:]
        img3 = img1
        # crop = transforms.RandomResizedCrop(31, interpolation=transforms.InterpolationMode.BICUBIC)
        # flip = transforms.RandomHorizontalFlip(0.5)
        tensor_trans = transforms.ToTensor()
        tn = transforms.Normalize(np.zeros(15), np.ones(15))
        img1 = tensor_trans(img1)
        img1 = tn(img1)
        # img1 = color_jitter(flip(crop(img1)))
        # img1 = tensor_trans(img1)
        # img1 = tn(img1)
        img2 = tensor_trans(img2)
        img2 = tn(img2)

        img3 = tensor_trans(img3)
        img3 = tn(img3)

        # img2 = color_jitter(flip(crop(img2)))
        # img2 = tensor_trans(img2)
        # img2 = tn(img2)
        targets = self.label[i]
        # img, targets = preprocess(img, targets)
        return img1, img2, img3, targets

    def __len__(self):
        return len(self.data)

class HsiDataset_tes(Dataset):
    def __init__(self, data, label):
        self.data = data.reshape(-1, 19, 19, 30)
        self.label = label
        # self.transform = transform
        self.classes = label.max() + 1

    def __getitem__(self, i):
        # img = self.data[i]
        # tensor_trans = transforms.ToTensor()
        # tn = transforms.Normalize(np.zeros(30), np.ones(30))
        # img = tensor_trans(img)
        # img = tn(img)
        # targets = self.label[i]
        img1 = self.data[i, :, :, :15]
        img2 = self.data[i, :, :, 15:]
        tensor_trans = transforms.ToTensor()
        tn = transforms.Normalize(np.zeros(15), np.ones(15))
        img1 = tensor_trans(img1)
        img1 = tn(img1)
        img2 = tensor_trans(img2)
        img2 = tn(img2)
        targets = self.label[i]
        return img1, img2, targets

    def __len__(self):
        return len(self.data)




def preprocess(X, y):

    X = np.reshape(X, (-1, X.shape[3], X.shape[1], X.shape[2]))
    X = X.astype(np.float32)
    X = torch.from_numpy(X)
    y = y.astype(int)
    y = torch.from_numpy(y)

    return  X, y


def random_flip(img, mode):
    """
    随机翻转
    :param img:
    :param model: 1=水平翻转 / 0=垂直 / -1=guangpu
    :return:
    """
    assert mode in (0, 1, -1), "mode is not right"
    flip = np.random.choice(2) * 2 - 1  # -1 / 1
    if mode == 1:
        img = img[:, ::flip, :]
    elif mode == 0:
        img = img[::flip, :, :]
    elif mode == -1:
        img = img[:, :, ::flip]

    return img

def random_noise(img, rand_range=(3, 20)):
    """
    随机噪声
    :param img:
    :param rand_range: (min, max)
    :return:
    """
    img = np.asarray(img, np.float)
    sigma = random.randint(*rand_range)
    nosie = np.random.normal(0, sigma, size=img.shape)
    img += nosie
    img = np.uint8(np.clip(img, 0, 255))
    return img