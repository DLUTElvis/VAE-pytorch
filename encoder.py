#_*_coding:utf-8_*_#
# Author: liao
# Date  : 2020/2/3 15:55
# FILE  : encoder.py
# IDE   : PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, D_in, H, D_out, CUDA):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)
        self.CUDA = CUDA
    def forward(self, x):

        if self.CUDA:
            x = x.cuda()
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return x
