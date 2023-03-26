# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from sklearn import metrics
from sklearn import decomposition
from sklearn import manifold
from tqdm.notebook import trange, tqdm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2

import copy
import random
import time

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.input_fc = nn.Linear(input_dim, 250)
        self.hidden_fc = nn.Linear(250, 100)
        self.output_fc = nn.Linear(100, output_dim)

    def forward(self, x):
        # x = [batch size, height, width]
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        # x = [batch size, height * width]
        h_1 = F.relu(self.input_fc(x))
        # h_1 = [batch size, 250]
        h_2 = F.relu(self.hidden_fc(h_1))
        # h_2 = [batch size, 100]
        y_pred = self.output_fc(h_2)
        # y_pred = [batch size, output dim]
        return y_pred, h_2

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32,64, kernel_size=5)
        self.fc1 = nn.Linear(3*3*64, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        #x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv3(x),2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.view(-1,3*3*64 )
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class DigitPredictor:
    def __init__(self, model_path, input_dim=None, output_dim=None):
        INPUT_DIM = 28 * 28
        OUTPUT_DIM = 10
        if input_dim is not None:
            INPUT_DIM = input_dim
            OUTPUT_DIM = output_dim
        self.model = MLP(INPUT_DIM, OUTPUT_DIM)
        # self.model = CNN()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.load_state_dict(torch.load(model_path))
        self.model = self.model.to(self.device)

        ROOT = '.data'
        train_data = datasets.MNIST(root=ROOT,train=True,download=True)
        mean = train_data.data.float().mean() / 255
        std = train_data.data.float().std() / 255
        train_transforms = transforms.Compose([
                            transforms.RandomRotation(5, fill=(0,)),
                            transforms.RandomCrop(28, padding=2),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[mean], std=[std])
                                      ])
        self.input_transforms = transforms.Compose([ transforms.ToTensor(), transforms.Normalize(mean=[mean], std=[std])])

    def prepare_input(self, input_im):
        # input_im is a 28*28 cv_image. Convert to np array
        # then transform to Tensor and Normalize
        n = np.asarray(input_im)
        tensor = self.input_transforms(n)
        return tensor

    def predict(self, input_im):
        input_im = self.prepare_input(input_im) # convert from cv image to tensor
        input_im = input_im.to(self.device)
        #y_pred, _ = self.model(input)

        with torch.no_grad():
            output, _ = self.model(input_im)
        pred = output.argmax().item()

        return pred

