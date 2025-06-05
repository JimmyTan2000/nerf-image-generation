# This repository is used to generate datasets for our Pose Estimation + Novel View Synthesis Practical using Nerf
The main general notebook is `nerf_train_general.ipynb`.

## Important notes:
The generated dataset is in .npz file which contains the tensor for the generated images, poses as well as focal length. In order to use it for our [SiT](https://github.com/JimmyTan2000/SiT), we convert those tensors into JPEG images for experimental purpose. Ideally we would want to feed in the tensors directly though to reduce some overhead during training. 

The python script for the JPEG conversion is `convert-to-jpg.py`

The generated image is 100 x 100 pixel. However, for SiT, the dimensions need to be divisible by 8 (eg: 128 x 128, 256 x 256, 512 x 512). In order to resize them, we can use the `resize.py` script. 