# Computation Imaging Group - MRI Reconstruction with DnCNN
Consider Magnetic Resonance Imaging (MRI) reconstruction as an inverse problem that recovers an image x from a noisy measurement y characterized by a linear model
y=PFx+e. 

The goal of this project was to implement a convolutional neural network (CNN) that maps noisy input images (F^(-1)y) to their corresponding ground-truth (x). 

The model I chose to use was a Denoising CNN (DnCNN). The original implementation was from _Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising_ and the Python implementation was from https://github.com/SaoYan/DnCNN-PyTorch.
