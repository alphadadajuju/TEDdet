# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Dequan Wang and Xingyi Zhou
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

from .deconv import deconv_layers

import matplotlib.pyplot as plt
import numpy as np

import torch.nn.functional as F

BN_MOMENTUM = 0.1


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, is_shift=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride
        
    # ADDED: debug feat map
    def vis_feat(self, x, t, c):
        
        x_np = x.cpu().detach().numpy()
        
        # x_np.shape: nt, c, h, w
        
        tar_feat = x_np[t, c, :,:] # shape: h, w
        plt.imshow(tar_feat)
        plt.title('Channel ' + str(c+1) + '| Time ' + str(t+1))
        plt.colorbar()
        plt.show()
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

    
resnet_spec = {18: (BasicBlock, [2, 2, 2, 2])}

class MOC_ResNet(nn.Module):
    def __init__(self, num_layers, n_segment=3, use_TD=False):
        super(MOC_ResNet, self).__init__()
        self.output_channel = 64
        block, layers = resnet_spec[num_layers]
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(BasicBlock, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, layers[3], stride=2)

        
        self.init_weights()
    
    
        
    def forward(self, input):
        x = self.conv1(input) # 144x144x64
        x = self.bn1(x)
        x = self.relu(x)
    
        x = self.maxpool(x)
        x = self.layer1(x) # 72x72x64
        x = self.layer2(x) # 36x36x128
        x = self.layer3(x) # 18x18x256
        x = self.layer4(x) # 9x9x512
        
        #x = self.deconv_layer(x)

        return x

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, is_shift=True))

        return nn.Sequential(*layers)

    def init_weights(self):
        pass
        # print('=> init deconv weights from normal distribution')
        #for name, m in self.deconv_layer.named_modules():
        #    if isinstance(m, nn.BatchNorm2d):
        #        nn.init.constant_(m.weight, 1)
        #        nn.init.constant_(m.bias, 0)
        
    