# TEDdet
Pytorch implementation of our IEEE publication "Temporal Feature Exchange and Difference Network for online real-time action detection" have been released.

## TEDdet Overview
we propose a lightweight action tubelet detector coined **TEDdet** which unifies complementary feature aggregation and motion modeling modules. Specifically, our Temporal Feature Exchange module facilitates feature interaction by aggregating action-specific visual patterns over successive frames, enabling spatiotemporal modeling on top of 2D CNN. To address actors' location shift in the sequence, our Temporal Feature Difference module approximates pair-wise motion among target frames in their abstract latent space. These modules can be easily integrated with an existing anchor-free detector (CenterNet) to cooperatively model action instances' categories, sizes and trajectories for precise tubelet generation. TEDdet exploits larger temporal strides to efficiently infer actions in a coarse-to-fine and online manner. 

* We present two lightweight temporal modeling modules: Temporal Feature Exchange (TE) and Temporal Feature Difference (TD) to facilitate learning action-specific spatiotemporal pattern and trajectory.

* We propose TEDdet, an integrated action tubelet detector on top of 2D CenterNet and TE-TD plug-in. Our detector operates in a coarse-to-fine manner. Alongside the online tube generation algorithm, TEDdet's detection speed well exceeds real-time requirement (89 FPS).

* Comprehensive analysis in terms of TEDdet's accuracy, robustness, and efficiency are conducted on public UCF-24 and JHMDB-21 datasets. Without relying on any 3D CNN or optical flow, our action detector achieves competitive accuracy at an unprecedented speed, suggesting a much more feasible solution pertinent to realistic applications.

## TEDdet Usage
### 1. Installation and Dataset
Please refer to https://github.com/MCG-NJU/MOC-Detector for detailed instructions.

### 2. Train
The current version of TEDdet support ResNet18 as the feature extraction backbone. To proceed with training, first download the COCO pretrained weights in [Google Drive](https://drive.google.com/drive/folders/1r2uYo-4hL6oOzRARFsYIn5Pu2Lv7VS6m). COCO pretrained models come from [CenterNet](https://github.com/xingyizhou/CenterNet). Move pretrained models to ```${TEDdet_ROOT}/experiment/modelzoo</mark>.```

To train your own model, run the ```train.py``` script along with relevant input arguments. For instance:
```python train.py --K 5 --exp_id K5_model --rgb_model TED_K5 --batch_size 16 --master_batch 16 --lr 2.5e-4 --gpus 0 --num_worker 16 --num_epochs 10 --lr_step 5 --dataset hmdb --split 1 --down_ratio 8 --lr_drop 0.1 --ninput 1 --ninputrgb 5 --auto_stop --pretrain coco ```

### 3. Evaluation

## References
