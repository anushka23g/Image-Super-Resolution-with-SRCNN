We propose a deep learning method for single image super-resolution (SR)
The aim of our project is to recover a high resolution image from a low resolution input.

To accomplish this goal, we will be deploying the super-resolution convolution neural network (SRCNN) using Keras.

Problem statement
SINGLE IMAGE SUPER RESOLUTION: Our Problem statement is the Enlarging an image with the Details recovered. To have an efficient output we need to take care of the following things:

1 Estimating the high frequency information that has been lost, such as the edges, texture ,etc
2 Severly ill-posed inverse problem
3 exploits the ontextual information as well
SR is an ill-posed problem because each LR pixel has to be mapped onto many HR pixels
it is an underdetermined inverse problem, of which solution is not unique

A lot of work has been done on this topic in past. Some of which is:
single-image super resolution algorithms can be categorized into four types – prediction models, edge based methods, image statistical methods and patch based (or example-based) methods

1 internal example based method
These methods either exploit internal similarities of the same image or learn mapping functions from external low- and high-resolution exemplar pairs

2 external example-based methods
based on a dictionary of low and highresolution exemplars

learn a mapping between low/high resolution patches from external datasets

Sparse Coding for Super Resolution
external example-based SR method

Sparse representation encodes a signal vector x as the linear combination of a few atoms in a dictionary D, i.e., x ≈ Dα, where α is the sparse coding vector

At first the overlapping patches present in the image are subtracted, the image is then normalised and pre-processed

These patches are then encoded by a low-resolution dictionary. The sparse coefficients are passed into a high-resolution dictionary for reconstructing high-resolution patches.

some mapping functions:

nearest neighbour

random forest

kernel regression
Our approach:
Our model does not explicitly learn the dictionaries for modeling the patch space

Rather to model the layers we use hidden layers as in a neural network

all the steps in sparse coding are performed by learning rather than pre-processing

It is fully feed -forward

STEPS
we first upscale it to the desired size using bicubic interpolation and obtain interpolated image as Y

x : high-resolution image
y : interpolated image low resolution
f(Y): reconstructed image
f is the mapping we do to find hr image from lr image

F consists of three operations:

Patch extraction and representation- extracts and represent overlapping patches as high-dimensional vector

Non-linear mapping: this operation nonlinearly maps each high-dimensional vector onto another high-dimensional vector.

Reconstruction:aggregates the above high-resolution patch-wise representations to generate the final high-resolution image

first layer of our model F1(Y) = max (0, W1 ∗ Y + B1)

W1 and B1 represent the filters and biases

second layer F2(Y) = max (0, W2 ∗ F1(Y) + B2).

Previously the overlapping patches thus found were averaged in the final reconstructed image

In our model, F(Y) = W3 ∗ F2(Y) + B3

hence all 3 layers can be taken as convolution layers



![different-layers](https://github.com/anushka23g/Image-Super-Resolution-with-SRCNN/blob/main/layers.png)


The Structural SIMilarity (SSIM) index is a method for measuring the similarity between two images.



![ssim-index](https://github.com/anushka23g/Image-Super-Resolution-with-SRCNN/blob/main/ssim.jpg)
