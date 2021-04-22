import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):

    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.l1 = nn.Linear(input_shape, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, output_shape)
    
    def forward(self, x, mask):
        y = F.relu(self.l1(x))
        y = F.relu(self.l2(y))
        y = F.relu(self.l3(y))
        return F.softmax(y) * mask


class ValueNetwork(nn.Module):

    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.l1 = nn.Linear(input_shape, 128)
        self.l2 = nn.Linear(128, 64)
        self.l3 = nn.Linear(64, output_shape)

    def forward(self, x):
        y = F.relu(self.l1(x))
        y = F.relu(self.l2(y))
        y = F.relu(self.l3(y))
        return y
