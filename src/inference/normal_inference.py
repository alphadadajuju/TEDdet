from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import numpy as np
from progress.bar import Bar
import torch
import pickle

from opts import opts
from datasets.init_dataset import switch_dataset
from detector.normal_moc_det import MOCDetector
import random

import time
import matplotlib.pyplot as plt
# MODIFY FOR PYTORCH 1+
# cv2.setNumThreads(0)
GLOBAL_SEED = 317


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def worker_init_fn(dump):
    set_seed(GLOBAL_SEED)


class PrefetchDataset(torch.utils.data.Dataset):
    def __init__(self, opt, dataset, pre_process_func):
        self.pre_process_func = pre_process_func
        self.opt = opt
        self.vlist = dataset._test_videos[dataset.split - 1]
     
        self.gttubes = dataset._gttubes
        self.nframes = dataset._nframes
        self.imagefile = dataset.imagefile
        self.flowfile = dataset.flowfile
        self.resolution = dataset._resolution
        self.input_h = dataset._resize_height
        self.input_w = dataset._resize_width
        self.output_h = self.input_h // self.opt.down_ratio
        self.output_w = self.input_w // self.opt.down_ratio
        self.indices = []
        
        # ADDED
        self.n_mem = self.opt.K - 1
        total_num_frames = 0
        
        for v in self.vlist:
            total_num_frames += self.nframes[v]
         
            if self.opt.rgb_model != '': 
                for i in range(min(self.opt.K * self.opt.ninputrgb , self.nframes[v]) - self.opt.ninputrgb + 1, 1 + self.nframes[v]):
                    if not os.path.exists(self.outfile(v, i)):
                        self.indices += [(v, i)]
                    
        print ('Finished loading det indices.')
        print ('There is a total of {} frames.'.format(total_num_frames))

    def __getitem__(self, index):
        
        v, frame = self.indices[index]
        h, w = self.resolution[v]
        images = []
        flows = []
        
        im_inds = []
        if self.opt.rgb_model != '' and self.opt.ninput == 1:

            n_mem =  self.n_mem
            im_inds = []
            
            # init frame of the sequence
            for _ in range(1): 
                im_inds.append(frame - 1)
                images.append(cv2.imread(self.imagefile(v, frame)).astype(np.float32))
            cur_f = frame
            
            # the rest of frames of the sequence
            for _ in range(1, n_mem+1):
                cur_f = np.maximum(cur_f - self.opt.ninputrgb, 1)
                im_inds.append(cur_f - 1)
                images.append(cv2.imread(self.imagefile(v, cur_f)).astype(np.float32))
            
            images.reverse() # time order: small to large
            im_inds.reverse()
            
            # debug 
            #print(im_inds)
            
            images = self.pre_process_func(images)
            
        outfile = self.outfile(v, frame)
        if not os.path.isdir(os.path.dirname(outfile)):
            os.system("mkdir -p '" + os.path.dirname(outfile) + "'")

        return {'outfile': outfile, 'images': images, 'flows': flows, 'meta': {'height': h, 'width': w, 'output_height': self.output_h, 'output_width': self.output_w}}

    def outfile(self, v, i):
        return os.path.join(self.opt.inference_dir, v, "{:0>5}.pkl".format(i))

    def __len__(self):
        return len(self.indices)

def vis_feat(image):
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
        

def normal_inference(opt, drop_last=False):
    # specify gpu id
    torch.cuda.set_device(opt.gpus[0])
    torch.backends.cudnn.benchmark = True

    Dataset = switch_dataset[opt.dataset]
    opt = opts().update_dataset(opt, Dataset)

    dataset = Dataset(opt, 'test') # test
    detector = MOCDetector(opt)
    prefetch_dataset = PrefetchDataset(opt, dataset, detector.pre_process) # check existing detection (skipping those that have been detected)
    total_num = len(prefetch_dataset)
    data_loader = torch.utils.data.DataLoader(
        prefetch_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=opt.pin_memory,
        drop_last=drop_last,
        worker_init_fn=worker_init_fn)

    num_iters = len(data_loader)

    bar = Bar(opt.exp_id, max=num_iters)

    print('inference chunk_sizes:', opt.chunk_sizes)
    print('Length of process data: {}'.format(len(data_loader)))
    
    data_time_start = time.time()
    data_time = 0
    for iter, data in enumerate(data_loader):
        
        '''
        # Debug: visualize input image
        #mmean = torch.from_numpy(np.array([0.40789654, 0.44719302, 0.47026115])).unsqueeze(1).unsqueeze(1)
        #sstd = torch.from_numpy(np.array([0.28863828, 0.27408164, 0.27809835])).unsqueeze(1).unsqueeze(1)
        #mmean = mmean.type(torch.FloatTensor)
        #sstd = sstd.type(torch.FloatTensor)
        for i in range(len(data['images'])):
            vis_feat(data['images'][i][0].permute(1,2,0))
            
            #plt.imshow(data['images'][i][0].permute(1,2,0).cpu())
            #plt.show()
        
        #((im_db * std[3*ii:3*(ii+1),:,:] + mean[3*ii:3*(ii+1),:,:]) * 255.)
        '''
        
        data_time_end = time.time()
        data_time += data_time_end - data_time_start
        
        outfile = data['outfile']
        
        detections, total_time = detector.run(data)
        
        if iter % 100 == 0:
            print('Data time {} seconds.'.format(data_time))
            print('Processed {}/{} frames; {} seconds.'.format(iter+1, num_iters, total_time))
        
        for i in range(len(outfile)):
            with open(outfile[i], 'wb') as file:
                pickle.dump(detections[i], file)

        Bar.suffix = 'inference: [{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
            iter, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
        bar.next()
        
        data_time_start = time.time()
        
    bar.finish()
        
    return total_num
