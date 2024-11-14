# Foundation of Generative Models

## Homework 2

Main contributor: 易小鱼
Teammates: 刘恺河、易鼎程

## Dataset and Dataloader

We use 12 subclasses of the [quick draw](https://github.com/googlecreativelab/quickdraw-dataset) dataset. In each subclass are 5000 hand-drawing images. All images are used for training for simplicity.
Each batch of training data contains a batch of images and a batch of corresponding labels. The former is of size [batch_size, 1, 255, 255], and the latter is of size [batch_size].

## Model Architecture

We use a classifier-free guidance diffusion model with U-Net as the backbone. 
