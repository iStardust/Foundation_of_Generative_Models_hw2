# Foundation of Generative Models

## Homework 2

Main contributor: 易小鱼
Teammates: 刘恺河、易鼎程

## Dataset and Dataloader

We use 12 subclasses of the [quick draw](https://github.com/googlecreativelab/quickdraw-dataset) dataset. In each subclass are 5000 hand-drawing images. All images are used for training for simplicity.
Each batch of training data contains a batch of images and a batch of corresponding labels. The former is of size [batch_size, 1, 255, 255], and the latter is of size [batch_size].

## Model Architecture

We use a classifier-free guidance diffusion model with U-Net as the backbone. Given a batch of training data including images $x$ and labels $c$, we randomly sample a batched diffusion timesteps $t$ and predicts the standard gaussian noise using the U-Net model $\epsilon_\theta(x,t,c)$ parameterized by $\theta$.

## Hyperparameters

All hyperparameters can be adjusted in `src\params.py`, or using command lines to specify some of them when running `python train.py`. We recommend using the latter approach since the model architecture is better unchanged.


## Usage

### Notice

If you are using Windows, it requires you to run this code using powershell in administrator mode.

### Training

Directly run `python train.py`. The code will automatically create a subfolder `outputs\default` for logging training loss and saving model checkpoints.

If you want to visualize the training loss along as the training goes, run `tensorboard --logdir=outputs`.




