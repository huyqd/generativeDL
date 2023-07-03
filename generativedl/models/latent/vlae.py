from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm


class CNNEncoder(nn.Module):
    def __init__(self, in_dim, latent_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.relu4 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(4 * 4 * 256, 2 * latent_dim)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out = self.conv4(out)
        out = self.relu4(out)
        out = self.flatten(out)
        mu, log_sigma = self.linear(out).chunk(2, dim=1)
        return mu, log_sigma


class CNNDecoder(nn.Module):
    def __init__(self, latent_dim, out_dim):
        super().__init__()
        self.linear1 = nn.Linear(latent_dim, 4 * 4 * 128)
        self.relu1 = nn.ReLU()
        self.transpose_conv1 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)
        self.relu2 = nn.ReLU()
        self.transpose_conv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.relu3 = nn.ReLU()
        self.transpose_conv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.relu4 = nn.ReLU()
        self.conv = nn.Conv2d(32, out_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu1(out)
        out = out.view(-1, 128, 4, 4)
        out = self.transpose_conv1(out)
        out = self.relu2(out)
        out = self.transpose_conv2(out)
        out = self.relu3(out)
        out = self.transpose_conv3(out)
        out = self.relu4(out)
        out = self.conv(out)
        return out


class CNNVAE(nn.Module):
    def __init__(self, in_dim, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = CNNEncoder(in_dim=in_dim, latent_dim=latent_dim)
        self.decoder = CNNDecoder(latent_dim=latent_dim, out_dim=in_dim)

    def loss(self, x):
        x = 2 * x - 1
        mu_z, log_std_z = self.encoder(x)
        z = torch.randn_like(mu_z) * log_std_z.exp() + mu_z
        x_hat = self.decoder(z)

        # Compute reconstruction loss - Note that it may be easier for you
        # to use torch.distributions.normal to compute the log_prob
        recon_loss = F.mse_loss(x_hat, x, reduction='none').view(x.shape[0], -1).sum(1).mean()

        # Compute KL
        kl_loss = -log_std_z - 0.5 + (torch.exp(2 * log_std_z) + mu_z ** 2) * 0.5
        kl_loss = kl_loss.sum(1).mean()

        return OrderedDict(
            loss=recon_loss + kl_loss, recon_loss=recon_loss, kl_loss=kl_loss
        )

    def sample(self, n, noise=True):
        with torch.no_grad():
            z = torch.randn(n, self.latent_dim).cuda()
            samples = torch.clamp(self.decoder(z), -1, 1)
        return samples.cpu().permute(0, 2, 3, 1).numpy() * 0.5 + 0.5
