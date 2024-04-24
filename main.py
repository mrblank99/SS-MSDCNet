import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import os
import torch
import pandas as pd
import argparse
import math
import h5py
import random
from torch.optim import SGD
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from sklearn.decomposition import PCA
import torch.nn.functional as F
from tqdm import tqdm
from operator import truediv
from xiaorong import Model
# from densnet import DenseNet
# from HybirdSN import HyperCLR
# from newmodel3D import DenseNet_3d
from newmodel import DenseNet_3d
# from MIFN import MM
# from trynew import DenseNet_3d
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from torch.utils.data import Dataset, DataLoader
from Hsidataset import HsiDataset_tra, HsiDataset_tes
from aguimg import augment




def train(net, data_loader, train_optimizer):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader, dynamic_ncols=True)
    for pos_1, pos_2, pos_3, target in train_bar:
        bsz = target.size(0)
        pos_1 = augment(pos_1)
        pos_2 = augment(pos_2)
        pos_1 = pos_1.unsqueeze(1)    #Newdense
        pos_2 = pos_2.unsqueeze(1)
        pos_3 = pos_3.unsqueeze(1)
        pos_1, pos_2, pos_3 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True), pos_3.cuda(non_blocking=True)
        feature_1, proj_1, _ = net(pos_1)
        feature_2, proj_2, _ = net(pos_2)
        _, _, outputs = net(pos_3)

        # compute simsiam loss
        sim_1 = -(F.normalize(proj_1, dim=-1) * F.normalize(feature_2.detach(), dim=-1)).sum(dim=-1).mean()
        sim_2 = -(F.normalize(proj_2, dim=-1) * F.normalize(feature_1.detach(), dim=-1)).sum(dim=-1).mean()
        loss_sim = 0.5 * sim_1 + 0.5 * sim_2

        # compute ctrr loss
        p1 = nn.functional.normalize(proj_1, dim=1)
        z2 = nn.functional.normalize(feature_2, dim=1)

        contrast_1 = torch.matmul(p1, z2.t())  # B X B

        # <q,z> + log(1-<q,z>)
        contrast_1 = -contrast_1 * torch.zeros(bsz, bsz).fill_diagonal_(1).cuda() + (
            (1 - contrast_1).log()) * torch.ones(bsz, bsz).fill_diagonal_(0).cuda()
        contrast_logits = 2 + contrast_1

        soft_targets = torch.softmax(outputs, dim=1)
        contrast_mask = torch.matmul(soft_targets, soft_targets.t()).clone().detach()
        contrast_mask.fill_diagonal_(1)
        pos_mask = (contrast_mask >= args.tau).float()
        contrast_mask = contrast_mask * pos_mask
        contrast_mask = contrast_mask / contrast_mask.sum(1, keepdim=True)
        loss_ctr = (contrast_logits * contrast_mask).sum(dim=1).mean(0)

        loss = loss_sim + args.lamb*loss_ctr


        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += pos_1.size(0)
        total_loss += loss.item() * pos_1.size(0)
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / total_num



def splitTrainTestSet(X, y, testRatio):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, stratify=y)
    return X_train, X_test, y_train, y_test

def feature_normalize(data):
    mu = np.mean(data,axis=0)
    std = np.std(data,axis=0)
    return truediv((data - mu),std)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
if __name__ == '__main__':
    dataset_names = ['IP', 'SA', 'PU']
    parser = argparse.ArgumentParser(description="Run deep learning experiments on"
                                                 " various hyperspectral datasets")
    parser.add_argument('--dataset', type=str, default='PU', choices=dataset_names,
                        help="Dataset to use.")
    parser.add_argument('--feature_dim', default=2048, type=int, help='Feature dim for latent vector')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--batch_size', default=128, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=150, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--lamb', default=8.0, type=float, help='lambda for contrastive regularization term')
    parser.add_argument('--tau', type=float, default=0.4, help='contrastive threshold (tau)')
    args = parser.parse_args()
    # TRAIN =args.train
    # epoch=args.epoch
    feature_dim, k = args.feature_dim, args.k
    batch_size, epochs = args.batch_size, args.epochs
    dataset = args.dataset
    bands = 30
    # perclass=args.perclass
    # perclass=perclass/100

    f = h5py.File('PU19-19-15.h5', 'r')
    data = f['data'][:]
    label = f['label'][:]
    f.close()

    print(label, label.max(), label.min(), label.shape)

    # data prepare
    train_data = HsiDataset_tra(data, label)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                              drop_last=True)


    model = DenseNet_3d(feature_dim=feature_dim, finetune=False).cuda()

    optimizer = SGD(model.parameters(), lr=0.03, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda i: 0.5 * (math.cos(i * math.pi / epochs) + 1))
    c = train_data.classes

    # training loop
    results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': []}
    save_name_pre = '{}_{}_{}_{}'.format(feature_dim, k, batch_size, epochs)
    if not os.path.exists('results'):
        os.mkdir('results')
    best_acc = 0.0

    # ############################################################################################
    # for epoch in range(1, epochs + 1):
    #     train_loss = train(model, train_loader, optimizer)
    #     lr_scheduler.step()
    # torch.save(model.state_dict(), 'results/{}_modelpuloss.pth'.format(save_name_pre))
    ############################################################################################

    model.load_state_dict(torch.load('results/2048_200_128_150_model3DdensenewSAHY19.pth'))

    f = h5py.File('PU19-19-15.h5', 'r')
    data = f['data'][:]
    label = f['label'][:]
    f.close()

    memory_data = HsiDataset_tes(data, label)
    memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # model.eval()
    feature_bank = []


    with torch.no_grad():
        for data, _, target in (memory_loader):
            data = data.unsqueeze(1)     #newdense
            # data2 = data2.unsqueeze(1)
            # print(data.shape, target.shape) # torch.Size([128, 3, 28, 28]) torch.Size([128])
            feature, out = model(data.cuda(non_blocking=True))
            # out1 = model(data1.cuda(non_blocking=True))
            # out2 = model(data2.cuda(non_blocking=True))
            # out = torch.cat([out1, out2], 1)
            # out = torch.add(out1, out2)
            feature_bank.append(out)
    # print()
    feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
    feature_labels = torch.tensor(memory_loader.dataset.label, device=feature_bank.device)
    x = feature_bank.cpu().numpy()
    y = feature_labels.cpu().numpy()
    x = x.T

    print(x.shape)
    print(y.shape)





    f = h5py.File('HY-featureaddnew.h5', 'w')
    f['data'] = x
    f['label'] = y
    f.close()



