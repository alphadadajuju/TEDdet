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
from detector.stream_moc_det import MOCDetector
import random

import time
import matplotlib.pyplot as plt 

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
    def __init__(self, opt, dataset, pre_process, pre_process_single_frame):
        self.pre_process = pre_process
        
        self.pre_process_single_frame = pre_process_single_frame
        
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
        
        self.n_mem = self.opt.K - 1
        
     
        total_num_frames = 0
   
        for v in self.vlist:
            
            total_num_frames += self.nframes[v]
            use_ind_flag = True # sample a frame when True
            ind_cd = self.opt.ninputrgb  
            
            for i in range(min(self.opt.K * self.opt.ninputrgb , self.nframes[v]) - self.opt.ninputrgb + 1, 1 + self.nframes[v]):
                
                if (use_ind_flag is True) or i == self.nframes[v]:
                        self.indices += [(v, i)]
                        use_ind_flag = False # to skip frames
                        ind_cd = self.opt.ninputrgb # restart count-down for frame sampling
                        
                ind_cd -= 1
                if ind_cd == 0: # sample the next frame
                    use_ind_flag = True
     
        print ('Finished loading det indices.')
        print ('There is a total of {} frames.'.format(total_num_frames))
        
        self.img_buffer = []
        
        self.img_buffer_flip = []
        
        self.last_video = -1 # indicates when swithing to a new video
        self.last_frame = -1 # previously processed frame
        
        # debug: to keep track of what frames actually being detected
        self.im_list_history = []
        
    def __getitem__(self, index):
        v, frame = self.indices[index]
        h, w = self.resolution[v]
        images = []
        flows = []
        video_tag = 0
        
        # if there is a new video
        # video_tag == 1: use buffer (when processing the same video)
        # video_tag == 0: process from scratch (upon processing a new video)
        
        if (v == self.last_video and frame == self.last_frame + self.opt.ninputrgb) or (v == self.last_video and frame == self.nframes[v]):
            video_tag = 1 
        else:
            video_tag = 0 

        self.last_video = v
        self.last_frame = frame
        
        if video_tag == 0:
            
            # clear out history for a fresh start
            self.im_list_history = []
            
            n_mem =  self.n_mem
            im_inds = []
            
            # for the init frame of the sequence
            for _ in range(1): 
                im_inds.append(frame - 1)
                images.append(cv2.imread(self.imagefile(v, frame)).astype(np.float32))
                self.im_list_history.append(frame)
   
            cur_f = frame
            
            # for the rest of frames of the sequence
            for _ in range(1, n_mem+1):
                cur_f = np.maximum(cur_f - self.opt.ninputrgb, 1)
                im_inds.append(cur_f - 1)
                images.append(cv2.imread(self.imagefile(v, cur_f)).astype(np.float32))
                self.im_list_history.append(cur_f)
            
            
            im_inds.reverse()
            images.reverse()
            self.im_list_history.reverse()
            
            #print ('num of frames of this vid: {}'.format(self.nframes[v]))
            #print(self.im_list_history)
            
            
            images = self.pre_process(images)
            self.img_buffer = images
        
        else:
             
            image = cv2.imread(self.imagefile(v, max(frame, 1))).astype(np.float32)
            self.im_list_history.append(frame)
            
            image, image_flip = self.pre_process_single_frame(image)
            
            del self.img_buffer[0] # FIFO: clear out the oldest feature
            self.img_buffer.append(image)
            
            images = self.img_buffer
        
        outfile = self.outfile(v, frame)
        if not os.path.isdir(os.path.dirname(outfile)):
            os.system("mkdir -p '" + os.path.dirname(outfile) + "'")

        return {'outfile': outfile, 'images': images, 'flows': flows, 'meta': {'height': h, 'width': w, 'output_height': self.output_h, 'output_width': self.output_w}, 'video_tag': video_tag}

    def outfile(self, v, i):
        return os.path.join(self.opt.inference_dir, v, "{:0>5}.pkl".format(i))

    def __len__(self):
        return len(self.indices)

def interpolate_detection(dets, list_update):
    
    raise NotImplementedError('Not implemented: detection interpolation in stream inference.')

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
        
def stream_inference(opt):
    torch.cuda.set_device(opt.gpus[0])
    # torch.backends.cudnn.benchmark = True

    Dataset = switch_dataset[opt.dataset]
    opt = opts().update_dataset(opt, Dataset)

    dataset = Dataset(opt, 'test')
    detector = MOCDetector(opt)
    prefetch_dataset = PrefetchDataset(opt, dataset, detector.pre_process, detector.pre_process_single_frame)
    data_loader = torch.utils.data.DataLoader(
        prefetch_dataset,
        batch_size=1, 
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        worker_init_fn=worker_init_fn)

    num_iters = len(data_loader)

    bar = Bar(opt.exp_id, max=num_iters)
    
    print('inference chunk_sizes:', opt.chunk_sizes)
    print('Length of process data: {}'.format(len(data_loader)))
    
    data_time_start = time.time()
    data_time = 0.0
    save_display_time = 0.0
    for iter, data in enumerate(data_loader):
        
        data_time_end = time.time()
        data_time += data_time_end - data_time_start
        
        outfile = data['outfile']
        detections, det_time = detector.run(data)
        
        if iter % 1000 == 0:
            print('Processed {}/{} frames.'.format(iter+1, num_iters))
                 
        # TODO: interpolation between frames
        # In fact, interp fits better in ACT_build
        #interpolate_detection(detections, data['K_frames'])
        
        save_display_start = time.time()
        
        for i in range(len(outfile)):
            with open(outfile[i], 'wb') as file:
                pickle.dump(detections[i], file)
        
        Bar.suffix = 'inference: [{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
            iter, num_iters, total=bar.elapsed_td, eta=bar.eta_td)

        bar.next()
        
        save_display_end = time.time()
        save_display_time += save_display_end - save_display_start 
        
        data_time_start = time.time()
    bar.finish()
    
    print('Processed all frames; data {} seconds.'.format(data_time))
    print('Processed all frames; det {} seconds.'.format(det_time))
    print('Processed all frames; save_display {} seconds.'.format(save_display_time))
    print('Processed all frames; total {} seconds.'.format(det_time + data_time + save_display_time))
