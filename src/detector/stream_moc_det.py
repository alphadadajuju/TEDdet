from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np

import torch

from MOC_utils.model import create_inference_model_rgb, load_inference_model_rgb
from .decode import moc_decode
from MOC_utils.utils import flip_tensor

import matplotlib.pyplot as plt

import time
class MOCDetector(object):
    def __init__(self, opt):
        if opt.gpus[0] >= 0:
            opt.device = torch.device('cuda')
        else:
            assert 'cpu is not supported!'

        self.rgb_model_backbone, self.rgb_model_deconv, self.rgb_model_branch = None, None, None
        
        if opt.rgb_model != '':
            self.rgb_model_backbone, self.rgb_model_deconv, self.rgb_model_branch = create_inference_model_rgb(opt, opt.arch, opt.branch_info, opt.head_conv, opt.K, flip_test=opt.flip_test)
            print('create rgb model', flush=True)
            self.rgb_model_backbone, self.rgb_model_deconv, self.rgb_model_branch = load_inference_model_rgb(self.rgb_model_backbone, self.rgb_model_deconv, self.rgb_model_branch, opt.save_root + opt.rgb_model)
            print('load rgb model', flush=True)
            self.rgb_model_backbone = self.rgb_model_backbone.to(opt.device)
            self.rgb_model_deconv = self.rgb_model_deconv.to(opt.device)
            self.rgb_model_branch = self.rgb_model_branch.to(opt.device)
            print('put rgb model to gpu', flush=True)
            self.rgb_model_backbone.eval()
            self.rgb_model_deconv.eval()
            self.rgb_model_branch.eval()
        
        self.num_classes = opt.num_classes
        self.opt = opt

        self.rgb_buffer = []
        
        # added: for speed measurement
        self.total_time = 0.0

    def pre_process(self, images, is_flow=False, ninput=1): # len(images): K*ninput (20)

        K = self.opt.K
        images = [cv2.resize(im, (self.opt.resize_height, self.opt.resize_width), interpolation=cv2.INTER_LINEAR) for im in images]

        if self.opt.flip_test:
            data = [np.empty((3 * ninput, self.opt.resize_height, self.opt.resize_width), dtype=np.float32) for i in range(K * 2)]
        else:
            data = [np.empty((3 * ninput, self.opt.resize_height, self.opt.resize_width), dtype=np.float32) for i in range(K)]

        mean = np.tile(np.array(self.opt.mean, dtype=np.float32)[:, None, None], (ninput, 1, 1))
        std = np.tile(np.array(self.opt.std, dtype=np.float32)[:, None, None], (ninput, 1, 1))

        for i in range(K):
            for ii in range(ninput):
                data[i][3 * ii:3 * ii + 3, :, :] = np.transpose(images[i*ninput + ii], (2, 0, 1)) # # added: *ninput
                if self.opt.flip_test:
                    # TODO
                    if is_flow:
                        temp = images[i + ii].copy()
                        temp = temp[:, ::-1, :]
                        temp[:, :, 2] = 255 - temp[:, :, 2]
                        data[i + K][3 * ii:3 * ii + 3, :, :] = np.transpose(temp, (2, 0, 1))
                    else:
                        data[i + K][3 * ii:3 * ii + 3, :, :] = np.transpose(images[i + ii], (2, 0, 1))[:, :, ::-1]
            # normalize
            data[i] = ((data[i] / 255.) - mean) / std
            if self.opt.flip_test:
                data[i + K] = ((data[i + K] / 255.) - mean) / std
                
        '''
        # DEBUG: visualize transformed images     
        #for i in range(K//2, K//2+1):
        for i in range(K):
            im_db_ = data[i]
            for ii in range(ninput):
                
                im_db = im_db_[3*ii:3*(ii+1), :,:]
                im_db = ((im_db * std[3*ii:3*(ii+1),:,:] + mean[3*ii:3*(ii+1),:,:]) * 255.)
                im_db = im_db.transpose(1,2,0)
                im_db = cv2.cvtColor(im_db, cv2.COLOR_BGR2RGB)
                plt.imshow(im_db.astype(np.uint8))
                plt.show()
        '''
        return data
    
    def pre_process_single_frame(self, images, is_flow=False, ninput=1, data_last=None, data_last_flip=None):
        images = cv2.resize(images, (self.opt.resize_height, self.opt.resize_width), interpolation=cv2.INTER_LINEAR)

        data = np.empty((3 * ninput, self.opt.resize_height, self.opt.resize_width), dtype=np.float32)
        data_flip = np.empty((3 * ninput, self.opt.resize_height, self.opt.resize_width), dtype=np.float32)

        mean = np.array(self.opt.mean, dtype=np.float32)[:, None, None]
        std = np.array(self.opt.std, dtype=np.float32)[:, None, None]
        if not is_flow:
            data = np.transpose(images, (2, 0, 1))
            if self.opt.flip_test:
                data_flip = np.transpose(images, (2, 0, 1))[:, :, ::-1]
            data = ((data / 255.) - mean) / std
            if self.opt.flip_test:
                data_flip = ((data_flip / 255.) - mean) / std

        else:
            data[:3 * ninput - 3, :, :] = data_last[3:, :, :]
            data[3 * ninput - 3:, :, :] = (np.transpose(images, (2, 0, 1)) / 255. - mean) / std
            if self.opt.flip_test:
                temp = images.copy()
                temp = temp[:, ::-1, :]
                temp[:, :, 2] = 255 - temp[:, :, 2]
                data_flip[:3 * ninput - 3, :, :] = data_last_flip[3:, :, :]
                data_flip[3 * ninput - 3:, :, :] = (np.transpose(temp, (2, 0, 1)) / 255. - mean) / std
        return data, data_flip

    def process(self, images, flows, video_tag): # video_tag == 0: new video no cach mechanism
        with torch.no_grad():
            if self.rgb_model_backbone is not None:
                if video_tag == 0:
                    
                    rgb_features = [self.rgb_model_backbone(images[i]) for i in range(self.opt.K)]
                    
                    
                    self.rgb_buffer = rgb_features
                    self.rgb_buffer = torch.cat(self.rgb_buffer, dim=0)
                    
                else:
                    
                    n_segment = self.rgb_buffer.shape[0]
                    self.rgb_buffer_deque, self.rgb_buffer = self.rgb_buffer.split([1, n_segment-1], dim=0)
                    
                    rgb_buffer_new = self.rgb_model_backbone(images[self.opt.K - 1])
                    self.rgb_buffer = torch.cat((self.rgb_buffer, rgb_buffer_new), dim=0)
                
                
                self.rgb_buffer_deconv = self.rgb_model_deconv(self.rgb_buffer)
                rgb_output = self.rgb_model_branch(self.rgb_buffer_deconv)
                
                #rgb_hm = rgb_output[0]['hm'].sigmoid_()
                rgb_hm = rgb_output[0]['hm']
                rgb_wh = rgb_output[0]['wh']
                rgb_mov = rgb_output[0]['mov']
            
            if self.rgb_model_backbone is not None:
                hm = rgb_hm
                wh = rgb_wh
                mov = rgb_mov
            
            else:
                print('No model exists.')
                assert 0

            detections = moc_decode(hm, wh, mov, N=self.opt.N, K=self.opt.K)
            return detections # size: (1, 100, 18): last dim == 4K + 1 + 1 (box, score, cls) 

    def post_process(self, detections, height, width, output_height, output_width, num_classes, K):
        detections = detections.detach().cpu().numpy()

        results = []
        for i in range(detections.shape[0]):
            top_preds = {}
            for j in range((detections.shape[2] - 2) // 2):
                # tailor bbox to prevent out of bounds
                detections[i, :, 2 * j] = np.maximum(0, np.minimum(width - 1, detections[i, :, 2 * j] / output_width * width))
                detections[i, :, 2 * j + 1] = np.maximum(0, np.minimum(height - 1, detections[i, :, 2 * j + 1] / output_height * height))
            classes = detections[i, :, -1]
            # gather bbox for each class
            for c in range(self.opt.num_classes):
                inds = (classes == c)
                top_preds[c + 1] = detections[i, inds, :4 * K + 1].astype(np.float32)
            results.append(top_preds)
            return results

    def run(self, data):

        flows = None
        images = None

        if self.rgb_model_backbone is not None:
            images = data['images']
            for i in range(len(images)):
                images[i] = images[i].to(self.opt.device)
        
        meta = data['meta']
        meta = {k: v.numpy()[0] for k, v in meta.items()}
        
        detection_start = time.time()
        
        detections = self.process(images, flows, data['video_tag']) # detections.size(): torch.Size([1, 100, 18])
        
        detections = self.post_process(detections, meta['height'], meta['width'],
                                       meta['output_height'], meta['output_width'],
                                       self.opt.num_classes, self.opt.K)
        detection_end = time.time()
        self.total_time += detection_end - detection_start
        return detections, self.total_time
