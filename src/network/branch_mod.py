from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import torch

from torch import nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

import time

import numpy as np
import random
import os

import math

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False    
    torch.backends.cudnn.deterministic = True
    
def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
  
class MOC_Branch(nn.Module):
    def __init__(self, input_channel, arch, head_conv, branch_info, K):
        super(MOC_Branch, self).__init__()
        assert head_conv > 0
        wh_head_conv = 64 if (arch == 'resnet' or arch == 'mobile_v2') else head_conv
        
     
        self.shrink = nn.Sequential(
            nn.Conv2d(input_channel, input_channel//4, 
                      kernel_size=1, padding=0, bias=False, groups=1),
            
            nn.BatchNorm2d(num_features=input_channel//4),
            nn.ReLU(inplace=True)
            )
        
        
        
        self.hm = nn.Sequential(
            nn.Conv2d(2*input_channel//4, head_conv, 
                      kernel_size=3, padding=1, bias=True, groups=1, dilation=1),
            nn.ReLU(inplace=True)
            )
        
        self.hm_cls = nn.Sequential(nn.Conv2d(head_conv, branch_info['hm'],
                                              kernel_size=1, stride=1,
                                              padding=0, bias=True, groups=1))
        
        
        self.hm_cls[-1].bias.data.fill_(-2.19) # -2.19
        
        
        self.mov = nn.Sequential(
            nn.Conv2d(K*input_channel//4, head_conv, 
                      kernel_size=3, padding=1, bias=True, groups=1, dilation=1),
            nn.ReLU(inplace=True))
        
        self.mov_cls = nn.Sequential(nn.Conv2d(head_conv, (branch_info['mov']), 
                      kernel_size=1, stride=1,
                      padding=0, bias=True))
        
        fill_fc_weights(self.mov)
        fill_fc_weights(self.mov_cls)
 
        #============================================================
        self.wh = nn.Sequential(
            nn.Conv2d(input_channel//4, wh_head_conv,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(wh_head_conv, branch_info['wh'] // K,
                      kernel_size=1, stride=1,
                      padding=0, bias=True))
        fill_fc_weights(self.wh)
       
        self.init_weights()
        
        # unidirectional shift LEFT and RIGHT
        
        if K == 3:
            ## two uni-directional TE
            self.shift = ShiftModule(input_channels=64, n_segment=K, n_div=4, mode='shift_right')
            self.shift_rev = ShiftModule(input_channels=64, n_segment=K, n_div=4, mode='shift_left')
            
            # one bidirectional TE
            #self.shift = ShiftModule(input_channels=64, n_segment=K, n_div=4, mode='shift')
            
            # baseline: no shift at all
            #pass
        elif K == 5: 
            self.shift = ShiftModule(input_channels=64, n_segment=K, n_div=3, mode='shift_right')
            self.shift_2 = ShiftModule(input_channels=64, n_segment=K, n_div=6, mode='shift_right')
            
            self.shift_rev = ShiftModule(input_channels=64, n_segment=K, n_div=3, mode='shift_left')
            self.shift_rev_2 = ShiftModule(input_channels=64, n_segment=K, n_div=6, mode='shift_left')
        
        self.me = MEModule(channel=64, reduction=8, n_segment=K)
        
        self.n_segment = K
        
    def init_weights(self):
       
        for name, m in self.shrink.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        
        for name, m in self.hm.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
       
        for name, m in self.hm_cls.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
            elif isinstance(m, nn.Conv2d):
                # normal distribution here gives a very strange spike / scale for the loss (due to different range of std?)
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', a=math.sqrt(5), nonlinearity='leaky_relu')
        
        
        for name, m in self.mov.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        
        for name, m in self.mov_cls.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
       
        for name, m in self.wh.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        
    def vis_feat(self, x, t, c):
        
        x_np = x.cpu().detach().numpy()
        
        # x_np.shape: nt, c, h, w
        
        tar_feat = np.abs(x_np[t, c, :,:]) # shape: h, w
        plt.imshow(tar_feat)
        plt.title('Channel ' + str(c+1) + '| Time ' + str(t+1))
        plt.colorbar()
        plt.show()
        
    def forward(self, input_chunk, K=3):
        
        bbK, cc, hh, ww = input_chunk.size()
        
        input_chunk_small = self.shrink(input_chunk)
        
        output = {}

        output_wh = (self.wh(input_chunk_small))
        output_wh = output_wh.view(bbK // K, -1, hh, ww)
        output['wh'] =  output_wh
        
        input_chunk_small_centeroffset = self.me(input_chunk_small)
        output_mov = self.mov(input_chunk_small_centeroffset.view(-1, cc*K//4, hh, ww))
        
        output['mov'] = self.mov_cls(output_mov)
        
        # unidirectional LEFT and RIGHT
        if self.n_segment == 3: 
            ## two uni-directional TE
            input_chunk_small_forward = self.shift(input_chunk_small)
            input_chunk_small_backward = self.shift_rev(input_chunk_small)
            
            # one bidirectional TE
            #input_chunk_small_forward = self.shift(input_chunk_small)
            
            # baseline single frame
            #input_chunk_small_forward = input_chunk_small
            
        elif self.n_segment == 5:
            input_chunk_small_forward = self.shift(input_chunk_small)
            input_chunk_small_forward = self.shift_2(input_chunk_small_forward)
            
            input_chunk_small_backward = self.shift_rev(input_chunk_small)
            input_chunk_small_backward = self.shift_rev_2(input_chunk_small_backward)
        
        input_chunk_small_forward_cent = input_chunk_small_forward.view(-1, K, cc//4, hh, ww)[:,K//2, :,:,:]
        input_chunk_small_backward_cent = input_chunk_small_backward.view(-1, K, cc//4, hh, ww)[:,K//2, :,:,:]
        
        input_chunk_small_forward_cent = torch.cat((input_chunk_small_forward_cent, input_chunk_small_backward_cent), dim=1)
        output_hm = self.hm(input_chunk_small_forward_cent)
        
        output_hm = self.hm_cls(output_hm)
        
        output['hm'] = output_hm.sigmoid_()
        
        return output
    
class MEModule(nn.Module):
    """ Motion exciation module
    
    :param reduction=16
    :param n_segment=8/16
    """
    def __init__(self, channel, reduction=16, n_segment=8):
        super(MEModule, self).__init__()
        self.channel = channel
        self.reduction = reduction
        self.n_segment = n_segment
        
        self.conv1 = nn.Conv2d(
            in_channels=self.channel,
            out_channels=self.channel//self.reduction,
            kernel_size=1,
            bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=self.channel//self.reduction)
        
        self.conv2 = nn.Conv2d(
            in_channels=self.channel//self.reduction,
            out_channels=self.channel//self.reduction,
            kernel_size=3,
            padding=1,
            groups=channel//self.reduction,
            bias=False)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

        self.pad = (0, 0, 0, 0, 0, 0, 0, 1)
        self.pad1_backward = (0, 0, 0, 0, 0, 0, 1, 0)
        
        self.conv3 = nn.Conv2d(
            in_channels=self.channel//self.reduction,
            out_channels=self.channel,
            kernel_size=1,
            bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=self.channel)
        
        #self.identity = nn.Identity()

    def forward(self, x):
        nt, c, h, w = x.size()
        bottleneck = self.conv1(x) # nt, c//r, h, w
        bottleneck = self.bn1(bottleneck) # nt, c//r, h, w

        # t feature
        reshape_bottleneck = bottleneck.view((-1, self.n_segment) + bottleneck.size()[1:])  # n, t, c//r, h, w
      
        # apply transformation conv to t+1 feature
        conv_bottleneck = self.conv2(bottleneck)  # nt, c//r, h, w
        # reshape fea: n, t, c//r, h, w
        reshape_conv_bottleneck = conv_bottleneck.view((-1, self.n_segment) + conv_bottleneck.size()[1:])
        
        ### forward (primary) direction
        tPlusone_fea = reshape_conv_bottleneck
        t_fea = reshape_bottleneck[:, self.n_segment//2, :, :].unsqueeze(dim=1)
        
        diff_fea = tPlusone_fea - t_fea # n, t-1, c//r, h, w
        diff_fea = diff_fea.view(nt, -1, h, w)
        
        y = self.conv3(diff_fea)  # nt, c, 1, 1
        y = self.bn3(y)  # nt, c, 1, 1
        
        return y
    
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
            self.conv.weight.data[self.fold:, 0, 1] = 1 # no shift 
            
            self.conv.weight.data[-1*self.fold:, 0, 0] = 1 # shift right
            self.conv.weight.data[-1*self.fold:, 0, 1] = 0 # shift right
            
        elif mode == 'shift_left':
            # import pdb; pdb.set_trace()
            self.conv.weight.requires_grad = True
            self.conv.weight.data.zero_()
            
            # orig: k = 3
            self.conv.weight.data[:2*self.fold, 0, 2] = 1 # fixed
            self.conv.weight.data[2*self.fold:, 0, 1] = 1 # shift left
            self.conv.weight.data = torch.flip(self.conv.weight, dims=[0,1])
          
        elif mode == 'shift_right':
            # import pdb; pdb.set_trace()
            self.conv.weight.requires_grad = True
            self.conv.weight.data.zero_()
            
            # orig: k = 3
            self.conv.weight.data[:2*self.fold, 0, 0] = 1 # shift right
            self.conv.weight.data[2*self.fold:, 0, 1] = 1 # fixed
            
       
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