import torch
import torch.nn as nn
from thop import profile, clever_format
import torchvision
import torch.nn.functional as F
from collections import OrderedDict


class SpatialSELayer3D(nn.Module):
    """
    3D extension of SE block -- squeezing spatially and exciting channel-wise described in:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018*
    """

    def __init__(self, num_channels=128):
        """
        :param num_channels: No of input channels

        """
        super(SpatialSELayer3D, self).__init__()
        self.conv = nn.Conv3d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor, weights=None):
        """
        :param weights: weights for few shot learning
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output_tensor
        """
        # channel squeeze
        batch_size, channel, D, H, W = input_tensor.size()

        if weights:
            weights = weights.view(1, channel, 1, 1)
            out = F.conv2d(input_tensor, weights)
        else:
            out = self.conv(input_tensor)

        squeeze_tensor = self.sigmoid(out)

        # spatial excitation
        output_tensor = torch.mul(input_tensor, squeeze_tensor.view(batch_size, 1, D, H, W))

        return output_tensor



class _DenseLayer_3d(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super().__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module(
            'conv1',
            nn.Conv3d(num_input_features,
                      bn_size * growth_rate,
                      kernel_size=1,
                      stride=1,
                      bias=False))
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module(
            'conv2',
            nn.Conv3d(bn_size * growth_rate,
                      growth_rate,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super().forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features,
                                     p=self.drop_rate,
                                     training=self.training)
        return torch.cat([x, new_features], 1)



class _DenseBlock_3d(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer_3d(num_input_features + i * growth_rate,
                                growth_rate, bn_size, drop_rate)
            self.add_module('denselayer{}'.format(i + 1), layer)

class _Transition_3d(nn.Sequential):

    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module(
            'conv',
            nn.Conv3d(num_input_features,
                      num_output_features,
                      kernel_size=1,
                      stride=1,
                      bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))

class Linear(nn.Module):
    def __init__(self, nb_classes=10, feat=512):
        super(Linear, self).__init__()
        self.linear = nn.Linear(feat, nb_classes)

    def forward(self, x):
        return self.linear(x)

class DenseNet_3d(nn.Module):
    """Densenet-BC model class
    Args:
        growth_rate (int) - how many filters to add each layer (k in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self,
                 growth_rate=32,
                 block_config=(6, 12, 24, 16),
                 num_init_features=64,
                 # num_init_features=24,
                 bn_size=4,
                 drop_rate=0,
                 num_classes=1000,
                 feature_dim=2048,
                 finetune=False):

        super().__init__()

        self.finetune = finetune
        self.SA = SpatialSELayer3D()

        # self.CA = ChannelAttention()


        self.features0 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv3d(1, num_init_features,
                                kernel_size=(7, 11, 11),   #原始

                                stride=(2, 2, 2),
                                padding=(3, 3, 3),
                                bias=False)),
            ('norm1', nn.BatchNorm3d(num_init_features)),
            ('relu1', nn.ReLU(inplace=True))]))

        self.features1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv3d(1, num_init_features,
                                kernel_size=(7, 9, 9),  #原始
                                stride=(2, 2, 2),
                                padding=(3, 3, 3),
                                bias=False)),
            ('norm1', nn.BatchNorm3d(num_init_features)),
            ('relu1', nn.ReLU(inplace=True))]))

        self.features2 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv3d(1, num_init_features,
                                kernel_size=(7, 7, 7),  #原始
                                stride=(2, 2, 2),
                                padding=(3, 3, 3),
                                bias=False)),
            ('norm1', nn.BatchNorm3d(num_init_features)),
            ('relu1', nn.ReLU(inplace=True))]))

        self.pool3D = nn.MaxPool3d(kernel_size=4, stride=1)
        self.pool3D_1 = nn.MaxPool3d(kernel_size=(4, 3, 3), stride=(1, 1, 1))  #7,5
        self.pool3D_2 = nn.MaxPool3d(kernel_size=(4, 2, 2), stride=(1, 1, 1))  # 3

        # Each denseblock
        num_features = num_init_features
        self.layer1 = _DenseBlock_3d(num_layers=block_config[0],
                                   num_input_features=num_features,
                                   bn_size=bn_size,
                                   growth_rate=growth_rate,
                                   drop_rate=drop_rate)
        num_features = num_features + block_config[0] * growth_rate
        self.transition1 = _Transition_3d(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        self.layer2 = _DenseBlock_3d(num_layers=block_config[1],
                                     num_input_features=num_features,
                                     bn_size=bn_size,
                                     growth_rate=growth_rate,
                                     drop_rate=drop_rate)
        num_features = num_features + block_config[1] * growth_rate
        self.transition2 = _Transition_3d(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        self.layer3 = _DenseBlock_3d(num_layers=block_config[2],
                                     num_input_features=num_features,
                                     bn_size=bn_size,
                                     growth_rate=growth_rate,
                                     drop_rate=drop_rate)
        num_features = num_features + block_config[2] * growth_rate
        self.transition3 = _Transition_3d(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        self.layer4 = _DenseBlock_3d(num_layers=block_config[3],
                                     num_input_features=num_features,
                                     bn_size=bn_size,
                                     growth_rate=growth_rate,
                                     drop_rate=drop_rate)
        num_features = num_features + block_config[3] * growth_rate



        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # projection head
        self.g = nn.Sequential(nn.Linear(384, 2048, bias=False), nn.BatchNorm1d(2048),
                               nn.ReLU(inplace=True), nn.Linear(2048, feature_dim, bias=False),
                               nn.BatchNorm1d(feature_dim))
        # self.g = projection_MLP(2048)
        # # prediction head
        self.h = nn.Sequential(nn.Linear(feature_dim, feature_dim // 4, bias=False), nn.BatchNorm1d(feature_dim // 4),
                               nn.ReLU(inplace=True), nn.Linear(feature_dim // 4, feature_dim, bias=True))

        self.linear = Linear(nb_classes=9, feat=384)


    def forward(self, x):


        x0 = x
        x1 = x[:, :, :, 2:19, 2:19]  #原始
        x2 = x[:, :, :, 3:18, 3:18]
        features0 = self.features0(x0)   #128*64*8*8*8
        out0 = self.layer1(features0)
        out0 = self.transition1(out0)
        out0 = self.SA(out0)
        out0 = self.pool3D(out0)

        features1 = self.features1(x1)
        out1 = self.layer1(features1)
        out1 = self.transition1(out1)
        out1 = self.SA(out1)
        out1 = self.pool3D(out1)

        features2 = self.features2(x2)  # 128*64*8*8*8
        out2 = self.layer1(features2)  # 128*256*8*8*8
        out2 = self.transition1(out2)
        out2 = self.SA(out2)
        out2 = self.pool3D(out2)

        out = torch.cat((out0, out1, out2), dim=1)
        out = torch.flatten(out, 1)

        output = self.linear(out)



        if self.finetune:
            return out
        else:
            feature = self.g(out)
            proj = self.h(feature)
        return feature, proj, output


def DenseNet1213D():
    return DenseNet_3d(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16))

if __name__=='__main__':
    model = DenseNet1213D()
    flops, params = profile(model, inputs=(torch.randn(1, 1, 15, 19, 19),))
    flops, params = clever_format([flops, params])
    print('# Model Params: {} FLOPs: {}'.format(params, flops))
    print(model)

    input = torch.randn(128, 1, 15, 19, 19)
    out, out1 = model(input)
    print(out.shape)
    print(out1.shape)

