import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

# import pytorch_lightning as pl

import torchvision
from torchvision.datasets import MNIST, FashionMNIST
from torchvision import transforms
from torchvision.utils import make_grid, save_image

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

import random
import os

class VAE(nn.Module):
  def __init__(self, in_channels=1, bottleneck=20, img_height=28, img_width=28):
    super(VAE, self).__init__()
    self.h = img_height
    self.w = img_width
    self.in_channels = in_channels
    self.device = "cpu"
    # encoder layers
    self.conv1 = nn.Conv2d(in_channels, 16, 3)
    self.conv2 = nn.Conv2d(16, 32, 3)
    # self.conv3 = nn.Conv2d(32, 64, 3)
    self.img_shape, self.conv_out_shape = self._out_size()
    # print(f"img shape {self.img_shape}")
    # print(f"out shape {self.conv_out_shape}")
    self.mean = nn.Linear(self.conv_out_shape.item(), bottleneck)
    self.logvar = nn.Linear(self.conv_out_shape.item(), bottleneck)

    # decoder layers
    # print(self.conv_out_shape)
    # print(bottleneck)
    self.dec_linear = nn.Linear(bottleneck, self.conv_out_shape.item(),)
    self.decConv1 = nn.ConvTranspose2d(32, 16, 3)
    self.decConv2 = nn.ConvTranspose2d(16, in_channels, 3)
    # self.decConv3 = nn.ConvTranspose2d(16, in_channels, 3)

  def _out_size(self):
    x = torch.zeros(1, self.in_channels, self.h, self.w).to(self.conv1.weight.device)
    out_shape = (self.conv2(self.conv1(x))).shape
    return out_shape, torch.prod(torch.tensor(out_shape))

  def enc_forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    # x = F.relu(self.conv3(x))
    x = x.view(x.shape[0], -1)
    mean = self.mean(x)
    logvar = self.logvar(x)
    return mean, logvar

  def dec_forward(self, x):
    x = F.relu(self.dec_linear(x))
    x = x.view(-1, self.img_shape[1], self.img_shape[2], self.img_shape[3])
    # print(x.shape)
    x = F.relu(self.decConv1(x))
    x = torch.sigmoid(self.decConv2(x))
    # x = torch.sigmoid(self.decConv3(x))
    # print(f"decoder output shape {x.shape}")
    return x

  def reparametrization(self, mu, logvar):
    eps = torch.randn_like(mu).to(self.device)
    return mu + eps * logvar.exp().pow(0.5)

  def forward(self, x):
    # pass the input through the encoder
    mu, logvar = self.enc_forward(x)
    # sample from the latent space 
    z = self.reparametrization(mu, logvar)
    # pass z through the decoder
    x_hat = self.dec_forward(z)
    return x_hat, mu, logvar

  def interpolate(self, x1, x2):
    if x1.dim() == 3:
      x1 = x1.unsqueeze(0)
    if x2.dim() == 3:
      x2 = x2.unsqueeze(0)
    if self.training:
      raise Exception(
          "Model is still in training mode"
      )
    mu1, logv1 = self.enc_forward(x1)
    mu2, logv2 = self.enc_forward(x2)
    z1 = self.reparametrization(mu1, logv1)
    z2 = self.reparametrization(mu2, logv2)
    weights = torch.arange(0.1, 0.9, 0.1)
    interpolated_images = [self.dec_forward(z1)]
    for weight in weights:
      interpolation = (1-weight) * z1 + weight * z2
      interpolated_images.append(self.dec_forward(interpolation))
    interpolated_images.append(self.dec_forward(z2))
    interpolated_images = torch.stack(interpolated_images, dim=0).squeeze(1)
    # print("shape")
    # print(interpolated_images.shape)
    return interpolated_images

