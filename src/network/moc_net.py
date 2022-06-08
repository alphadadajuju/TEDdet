from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch 
from torch import nn

from .branch_mod import MOC_Branch

from .resnet import MOC_ResNet
from .deconv import deconv_layers

import numpy as np
import cv2
import matplotlib.pyplot as plt 

import torch.nn.functional as F

import math

from einops import rearrange, repeat

backbone = {
    'resnet': MOC_ResNet # MOC_ResNet
}

def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]
        

class MOC_Net(nn.Module):
    def __init__(self, opt, arch, num_layers, branch_info, head_conv, K, flip_test=False):
        super(MOC_Net, self).__init__()
        self.flip_test = flip_test
        self.K = K
        
        self.backbone = backbone[arch](num_layers, K, opt.use_TD_in_backbone) if 'resnet' in opt.arch else backbone[arch](num_layers)
        self.deconv_layer = deconv_layers(inplanes=512, BN_MOMENTUM=0.1)
        self.branch = MOC_Branch(256, arch, head_conv, branch_info, K)
        
        self.shift = ShiftModule(input_channels=512, n_segment=K, n_div=8, mode='shift')
    
        self.init_weights()
        
    def forward(self, input):
        if self.flip_test:
            assert(self.K == len(input) // 2)
            chunk1 = [self.backbone(input[i]) for i in range(self.K)]
            chunk2 = [self.backbone(input[i + self.K]) for i in range(self.K)]

            return [self.branch(chunk1), self.branch(chunk2)]
        else:
            # sequentially; ORIG: MOC concept without long range mem
            #chunk = [self.backbone(input[i]) for i in range(self.K)]
            
            # TODO: alternative: parallel processing (squeeze into batch dim)   
            bb, cc, hh, ww = input[0].size()
            input_all = torch.cat(input, dim=1)
            input_all = input_all.view(-1, cc, hh, ww)
            
            '''
            # debug: original image
            ninput = input_all.size()[1] // 3
            for ii in range(self.K):
                for i in range(ninput):
                    self.vis_feat(input_all[ii:ii+1,i*3:i*3+3,:,:].cpu())
            '''
            
            chunk = self.backbone(input_all)
            
            '''
            # debug: image
            for ii in range(self.K):
                for i in range(input_all.size()[1]):
                    self.vis_feat(input_all[ii:ii+1,i,:,:].cpu())
            '''
            
            chunk = self.shift(chunk)
            chunk = self.deconv_layer(chunk)
            
            # higher-resolution feat map (for distinguishing smaller, overlapped targets)
            chunk = F.interpolate(chunk, [36, 36])
            
            return [self.branch(chunk, self.K)]
            
    # ADDED: to separate deconv layer (??)
    def init_weights(self):
        # print('=> init deconv weights from normal distribution')
        for name, m in self.deconv_layer.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def vis_feat(self, image):
        # ADDED: vis for debug
        # data[i] = ((data[i] / 255.) - mean) / std
        if image.size()[1] == 3:
            image_temp = image.numpy().squeeze().transpose(1,2,0)
            image_temp = ((image_temp * [0.28863828, 0.27408164, 0.27809835] + [0.40789654, 0.44719302, 0.47026115]) * 255).astype(np.uint8)
            image_temp = cv2.cvtColor(image_temp, cv2.COLOR_BGR2RGB)
        else: 
            image_temp = image.numpy().squeeze().astype(np.float32)
        plt.imshow(image_temp)
        plt.show()   
    
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