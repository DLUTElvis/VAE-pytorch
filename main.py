#_*_coding:utf-8_*_#
# Author: liao
# Date  : 2020/2/3 16:48
# FILE  : main.py
# IDE   : PyCharm
import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
from torchvision import transforms
import torch.utils.data as tud
from encoder import Encoder
from decoder import Decoder
from vae import VAE
import matplotlib.pyplot as plt
import numpy as np

def latent_loss(z_mean, z_stddev):
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev
    return -0.5 * torch.mean(1 + torch.log(stddev_sq) - mean_sq - stddev_sq)

if __name__ == '__main__':
    CUDA = torch.cuda.is_available()
    device = torch.device('cuda' if CUDA else 'cpu')
    input_dim = 28 * 28
    batch_size = 32
    inter_dim = 100
    latent_dim = 8
    transform = transforms.Compose([transforms.ToTensor()])
    mnist = torchvision.datasets.MNIST('./', download=True, transform=transform)
    dataloader = tud.DataLoader(mnist, batch_size=batch_size, shuffle=True)
    print('Number of samples:', len(mnist))
    encoder = Encoder(input_dim, inter_dim, inter_dim, CUDA)
    decoder = Decoder(latent_dim, inter_dim, input_dim, CUDA)
    if CUDA:
        vae = VAE(encoder, decoder, inter_dim, latent_dim, CUDA).to(device)
    else:
        vae = VAE(encoder, decoder, inter_dim, latent_dim, CUDA)

    criterion = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(vae.parameters(), lr=0.0001)
    l = 0
    for epoch in range(100):
        for i,data in enumerate(dataloader, 0):
            inputs, classes = data
            # inputs = Variable(inputs.resize_(batch_size, input_dim))
            # classes = Variable(classes)
            inputs.resize_(batch_size, input_dim)
            optimizer.zero_grad()
            if CUDA:
                inputs = inputs.to(device)
            dec = vae(inputs)
            l_latent = latent_loss(vae.z_mean,vae.z_stddev)
            l_recon = criterion(dec, inputs)
            loss = l_latent + l_recon
            loss.backward()
            optimizer.step()
            l = loss.item()
        print(epoch, l)
    if CUDA:
        plt.imshow(vae(inputs).data[0].cpu().numpy().reshape(28, 28), cmap='gray')
    else:
        plt.imshow(vae(inputs).data[0].numpy().reshape(28,28), cmap='gray')
    plt.show(block=True)


