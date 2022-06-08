import torch
from torch import nn
import math

import numpy as np
import os
import random

import torch.nn.functional as F

BN_MOMENTUM = 0.1

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False    
    torch.backends.cudnn.deterministic = True


class MM(nn.Module):
    def __init__(self, n_length=5, mm_mode='mm1'):
        super(MM, self).__init__()
    
        log_filter = self.create_log(sigma=1.1, size=5)
        
        self.spatial_conv = nn.Conv3d(in_channels=3,out_channels=3,kernel_size=(1,5,5),stride=(1,1,1),padding=(0,2,2), groups=1)
        #self.temp_conv = nn.Conv3d(in_channels=3,out_channels=3,kernel_size=(5,1,1),stride=(5,1,1),padding=(0,0,0), groups=3)
        
        self.spatial_conv.weight.requires_grad = True
        self.spatial_conv.weight.data.zero_()
        self.spatial_conv.weight.data[:,:,:,:,:] = torch.from_numpy(log_filter).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(3,3,1,1,1)
        
        #self.temp_conv.weight.requires_grad = True
        #self.temp_conv.weight.data.zero_()
        #self.temp_conv.weight.data[:,:,:,:,:] = torch.from_numpy(np.array([-1/10, -2/10, 4/10, -2/10, -1/10])).unsqueeze(1).unsqueeze(2)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
    def forward(self, x):
        
        # two things to change here:
        # 1. temporal none-channel-wise(?)
        # 2. 1,3,5 frames
        # 3. kernel center or right? (if right, then it is not an edge)
        
        x_rgb = x[:,-3:, :,:]
        h, w = x.size(-2), x.size(-1) 
        
        x = self.maxpool(x)
        x = x.view((-1, 3, 5) + x.size()[-2:])
        
        mm = self.spatial_conv(x)
        #mm = self.temp_conv(x)
        
        
        
        mm = mm.squeeze(2)
        mm = F.interpolate(mm, [h,w])
        
        return mm, x_rgb
    
    def l_o_g(self, x, y, sigma):
        # Formatted this way for readability
        nom = ( (y**2)+(x**2)-2*(sigma**2) )
        denom = ( (2*math.pi*(sigma**6) ))
        expo = math.exp( -((x**2)+(y**2))/(2*(sigma**2)) )
        return nom*expo/denom

    def create_log(self, sigma, size = 7):
        w = math.ceil(float(size)*float(sigma))
    
        if(w%2 == 0):
            w = w + 1
    
        l_o_g_mask = []
    
        w_range = int(math.floor(w/2))
        print("Going from " + str(-w_range) + " to " + str(w_range))
        for i in range(-w_range, w_range):
            for j in range(-w_range, w_range):
                l_o_g_mask.append(self.l_o_g(i,j,sigma))
        l_o_g_mask = np.array(l_o_g_mask)
        
        l_o_g_mask = l_o_g_mask.reshape(size+1,size+1)
        l_o_g_mask = l_o_g_mask[1:,1:]
        return l_o_g_mask
    
class MM_PA(nn.Module):
    def __init__(self, n_length=5, mm_mode='mm1'):
        super(MM_PA, self).__init__()
        
        self.mm_mode = mm_mode
        
        if self.mm_mode == 'mm1':
            self.shallow_conv = nn.Conv2d(in_channels=3,out_channels=8,kernel_size=5,stride=1,padding=2)
        
        
        
        elif self.mm_mode == 'mm2':
            self.shallow_conv = nn.Sequential(
                            nn.Conv2d(in_channels=3, out_channels=8, 
                                      kernel_size=5, padding=2),
                            #nn.BatchNorm2d(8, momentum=BN_MOMENTUM),
                            nn.ReLU(inplace=True),
                            #nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                            nn.Conv2d(in_channels=8, out_channels=8, 
                                  kernel_size=5, padding=2)
                )
        
        #self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0) # max pool before / kerne
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # max pool before / kerne
        
        self.n_length = n_length
        
        
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0, 0.001)
                nn.init.constant_(m.bias.data, 0)
        '''
        
    def forward(self, x): # x.size(): 16, 15, 288, 288
        x_rgb = x[:,-3:, :,:]
    
        h, w = x.size(-2), x.size(-1) 
        x = x.view((-1, 3) + x.size()[-2:])
        
        # added: downsample by 2 then upsample by 2 (to save FLOP)
        x_small = self.maxpool(x)
        hs, ws = x_small.size(-2), x_small.size(-1) # // 2 only for pa depth = 2? (an additional step of maxpool)
        
        if self.mm_mode == 'mm0':
            #pass
            x = x_small # if taking rgb diff
        
        elif self.mm_mode == 'mm1' or self.mm_mode == 'mm2':
            x = self.shallow_conv(x_small) # if taking 1 or 2 layer 
            #x = self.shallow_conv(x) # if taking 1 or 2 layer
        
        x = x.view(-1, self.n_length, x.size(-3), x.size(-2)*x.size(-1)) # torch.Size([16, 5, 8, 82944])
        for i in range(self.n_length-2):
            # True pairwise
            #d_i = nn.PairwiseDistance(p=2)(x[:,i,:,:], x[:,i+1,:,:]).unsqueeze(1) # torch.Size([16, 1, 82944])
            
            # TARGET - OTHERS
            d_i = nn.PairwiseDistance(p=2)(x[:,i,:,:], x[:,-1,:,:]).unsqueeze(1) # torch.Size([16, 1, 82944])
            #d_i = (torch.clamp(d_i, min=1e-2) - 1e-2) * 1.1
            
            
            # distance + relu (try to keep direction info?)
            #d_if = self.relu(x[:,-1,:,:] - x[:,i,:,:]).norm(p=2, dim=1, keepdim=True)
            #d_ib = self.relu(x[:,i,:,:] - x[:,-1,:,:]).norm(p=2, dim=1, keepdim=True)
            #d_i = d_i + d_ib #torch.cat((d_if, d_ib), 1)
            # no l2 norm?
            #d_i = x[:,i+1,:,:] - x[:,i,:,:]
            
            d = d_i if i == 0 else torch.cat((d, d_i), 1)
        
        
        mm = d.view(-1, 1*(self.n_length-2), hs, ws) # torch.Size([16, 4, 288, 288]) # PA = d.view(-1, 1*(self.n_length-2), hs, ws)
        mm = F.interpolate(mm, [h,w])
        
        return mm, x_rgb