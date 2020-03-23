import math
import torch
import torch.nn as nn
import torch.nn.functional as func


class Loss(nn.Module):
    loss = ['loss']

    def forward(self, feature, batch, train=False):
        raise NotImplementedError


class CrossEntropy(Loss):
    loss = ["loss"]

    def __init__(self, config):
        super(CrossEntropy, self).__init__()
        self.config = config

    def forward(self, feature, batch, train=False):
        label = batch.label
        return {'loss': func.cross_entropy(feature, label, reduction='mean')}




