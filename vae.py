#_*_coding:utf-8_*_#
# Author: liao
# Date  : 2020/2/3 15:48
# FILE  : vae.py
# IDE   : PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

class VAE(nn.Module):
    def __init__(self, encoder, decoder, encoder_out_dim, latent_dim, CUDA):
        super(VAE,self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_out_dim = encoder_out_dim
        self.latent_dim = latent_dim
        self.enc_mu = nn.Linear(self.encoder_out_dim, self.latent_dim)
        self.enc_log_sigma = nn.Linear(self.encoder_out_dim, self.latent_dim)
        self.CUDA = CUDA

    def _reparameter(self, h_enc):
        mu = self.enc_mu(h_enc)
        log_sigma = self.enc_log_sigma(h_enc)
        sigma = torch.exp(log_sigma)
        if self.CUDA:
            emp = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float().cuda()
        else:
            emp = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()#!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.z_mean = mu
        self.z_stddev = sigma
        z = mu + Variable(emp) * sigma
        return z

    def forward(self, input):
        input.cuda()
        self.inter = self.encoder(input)
        self.z = self._reparameter(self.inter)
        self.output = self.decoder(self.z)
        return self.output