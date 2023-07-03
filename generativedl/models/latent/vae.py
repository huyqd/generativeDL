from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNEncoder(nn.Module):
    def __init__(self, in_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(*[
            nn.Conv2d(in_dim, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4 * 4 * 256, 2 * latent_dim),
        ])

    def forward(self, x):
        mu, log_sigma = self.net(out).chunk(2, dim=1)
        return mu, log_sigma


class CNNDecoder(nn.Module):
    def __init__(self, latent_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(*[
            nn.Linear(latent_dim, 4 * 4 * 128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, out_dim, kernel_size=3, stride=1, padding=1),
        ])

    def forward(self, x):
        return self.net(x)


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
