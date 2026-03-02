import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from torch import optim

import torchvision
from torchvision.transforms.functional import pil_to_tensor
from torchvision.transforms import ToPILImage
from torchvision import transforms

from torchvision import  datasets
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np
import os
import tqdm 
from PIL import Image


classes = ["angry","disgust","fear","happy","neutral","sad", "surprise"]

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.c1 = nn.Conv2d(1, 16, 5)
    self.c2 = nn.Conv2d(16, 16, 5)

    self.s1 = nn.MaxPool2d(kernel_size=2, stride=2)

    self.c3 = nn.Conv2d(16,32,5)
    self.c4 = nn.Conv2d(32,32,5)

    self.s2 = nn.MaxPool2d(kernel_size=2, stride=2)

    self.fc1 = nn.Linear(1152,128)
    self.fc2 = nn.Linear(128,7)


  def forward(self, x):
    x = F.relu(self.c1(x))
    x = F.relu(self.c2(x))

    x = F.relu(self.s1(x))

    x = F.relu(self.c3(x))
    x = F.relu(self.c4(x))

    x = F.relu(self.s2(x))

    x = torch.flatten(x, 1)

    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x


