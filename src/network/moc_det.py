from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch import nn
#from .branch import MOC_Branch
from .branch_mod import MOC_Branch
#from .dla import MOC_DLA
from .resnet import MOC_ResNet
#from .MobileNetV2 import MobileNetV2, TDN_MobileNetV2

from .deconv import deconv_layers

import torch.nn.functional as F
import matplotlib.pyplot as plt 
import numpy as np
import cv2

backbone = {
    #'dla': MOC_DLA,
    'resnet': MOC_ResNet
}

    
class MOC_Backbone(nn.Module):
    def __init__(self, arch, num_layers,):
        super(MOC_Backbone, self).__init__()
        self.backbone = backbone[arch](num_layers)
        
    def forward(self, input):
        return self.backbone(input)
    
class MOC_Deconv(nn.Module):
    def __init__(self, inplanes, BN_MOMENTUM, K):
        super(MOC_Deconv, self).__init__()
        
        
        self.shift = ShiftModule(input_channels=512, n_segment=K, n_div=8, mode='shift')
        
        self.deconv_layer = deconv_layers(inplanes=512, BN_MOMENTUM=0.1)
        #self.init_weights() 
    
    def forward(self, input):
        
        
        input = self.shift(input)
        input = self.deconv_layer(input)
        
        output = F.interpolate(input, [36, 36])
        
        return output
    
    # ADDED: to separate deconv layer
    def init_weights(self):
        # print('=> init deconv weights from normal distribution')
        for name, m in self.deconv_layer.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
class MOC_Det(nn.Module):
    def __init__(self, backbone, branch_info, arch, head_conv, K, flip_test=False):
        super(MOC_Det, self).__init__()
        self.flip_test = flip_test
        self.K = K
        self.branch = MOC_Branch(256, arch, head_conv, branch_info, K) # backbone.backbone.output_channel == 64

    def forward(self, chunk1):
        assert(self.K == len(chunk1))
        
        return [self.branch(chunk1, self.K)]

class ShiftModule(nn.Module):
    """1D Temporal convolutions, the convs are initialized to act as the "Part shift" layer
    """

    def __init__(self, input_channels, n_segment=8, n_div=8, mode='shift'):
        super(ShiftModule, self).__init__()
        self.input_channels = input_channels
        self.n_segment = n_segment
        self.fold_div = n_div
        self.fold = self.input_channels // self.fold_div
        self.conv = nn.Conv1d(
            input_channels, input_channels,
            kernel_size=3, padding=1, groups=input_channels,
            bias=False)
        # weight_size: (2*self.fold, 1, 3)
        if mode == 'shift':
            # import pdb; pdb.set_trace()
            self.conv.weight.requires_grad = True
            self.conv.weight.data.zero_()
            self.conv.weight.data[:self.fold, 0, 2] = 1 # shift left
            self.conv.weight.data[self.fold: 2 * self.fold, 0, 0] = 1 # shift right
            if 2*self.fold < self.input_channels:
                self.conv.weight.data[2 * self.fold:, 0, 1] = 1 # fixed


    def forward(self, x):
        # shift by conv
        # import pdb; pdb.set_trace()
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)
        
        x = x.permute([0, 3, 4, 2, 1])  # (n_batch, h, w, c, n_segment)
        x = x.contiguous().view(n_batch*h*w, c, self.n_segment+0)
        x = self.conv(x)  # (n_batch*h*w, c, n_segment)
        x = x.view(n_batch, h, w, c, self.n_segment+0)
        x = x.permute([0, 4, 3, 1, 2])  # (n_batch, n_segment, c, h, w)
        
        x = x.contiguous().view(nt, c, h, w)
        return x
