# ReverseNeural3D
## Training Strategy
![image](https://user-images.githubusercontent.com/9930070/215598676-14d2420c-f206-4e8e-bac0-94a3a2d5cf02.png)

The input RGBD image is processed to 8 masked images corresponding to 8 target planes. The 8 masked images are then passed to the Reverse 3d Network, which predicts a phase to be displayed on SLM. During the training phase, the phase-only representation would be propagated back to 8 target planes with the trained forward CNNpropCNN model, where the loss can be calculated against the 8 masked target amplitudes.

## Files Description
- `load_image.py:` Load rgb and depth image, convert them into masked images
- `train.py:` Train reverse neural3d network
- `reverse3d_prop.py:` Define reverse neural3d architecture

## Dataset
- https://rgbd-dataset.cs.washington.edu/dataset/rgbd-scenes-v2/
