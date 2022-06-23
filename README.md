# TEDdet
Pytorch implementation of our IEEE publication "Temporal Feature Exchange and Difference Network for online real-time action detection" have been released.

## TEDdet Overview
we propose a lightweight action tubelet detector coined **TEDdet** which unifies complementary feature aggregation and motion modeling modules. Specifically, our Temporal Feature Exchange module facilitates feature interaction by aggregating action-specific visual patterns over successive frames, enabling spatiotemporal modeling on top of 2D CNN. To address actors' location shift in the sequence, our Temporal Feature Difference module approximates pair-wise motion among target frames in their abstract latent space. These modules can be easily integrated with an existing anchor-free detector (CenterNet) to cooperatively model action instances' categories, sizes and trajectories for precise tubelet generation. TEDdet exploits larger temporal strides to efficiently infer actions in a coarse-to-fine and online manner. 

* We present two lightweight temporal modeling modules: Temporal Feature Exchange (TE) and Temporal Feature Difference (TD) to facilitate learning action-specific spatiotemporal pattern and trajectory.

* We propose TEDdet, an integrated action tubelet detector on top of 2D CenterNet and TE-TD plug-in. Our detector operates in a coarse-to-fine manner. Alongside the online tube generation algorithm, TEDdet's detection speed well exceeds real-time requirement (89 FPS).

* Comprehensive analysis in terms of TEDdet's accuracy, robustness, and efficiency are conducted on public UCF-24 and JHMDB-21 datasets. Without relying on any 3D CNN or optical flow, our action detector achieves competitive accuracy at an unprecedented speed, suggesting a much more feasible solution pertinent to realistic applications.

## TEDdet Usage

## References
