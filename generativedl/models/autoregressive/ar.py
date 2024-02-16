import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Histogram(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.logits = nn.Parameter(torch.zeros(d), requires_grad=True)

    def loss(self, x):
        logits = self.logits.unsqueeze(0).repeat(x.shape[0], 1)  # batch_size x d
        return F.cross_entropy(logits, x.long())

    def get_distribution(self):
        distribution = F.softmax(self.logits, dim=0)
        return distribution.detach().cpu().numpy()


class MixtureOfLogistics(nn.Module):
    def __init__(self, d, n_mixture=4):
        super().__init__()
        self.d = d
        self.n_mixture = n_mixture

        self.pi = nn.Parameter(torch.randn(n_mixture), requires_grad=True)
        self.mu = nn.Parameter(torch.randn(n_mixture), requires_grad=True)
        self.log_sigma = nn.Parameter(torch.randn(n_mixture), requires_grad=True)

    def forward(self, x):
        x_rp = x.unsqueeze(1).repeat(1, self.n_mixture).float()
        pi = self.pi.unsqueeze(0)
        mu = self.mu.unsqueeze(0)
        inv_sigma = torch.exp(-self.log_sigma.unsqueeze(0))

        # probability dist for x in (0, d-1)
        cdf_plus = torch.sigmoid(inv_sigma * (x_rp + 0.5 - mu))
        cdf_minus = torch.sigmoid(inv_sigma * (x_rp - 0.5 - mu))
        cdf_delta = cdf_plus - cdf_minus

        # probability dist for x = 0: taking all value from -inf -> 0
        log_cdf_0 = torch.log(torch.clamp(F.sigmoid(inv_sigma * (0.5 - mu)), min=1e-12))
        # probability dist for x = 0: taking all value from -inf -> 0
        log_cdf_d_1 = torch.log(torch.clamp(1 - F.sigmoid(inv_sigma * (self.d - 1.5 - mu)), min=1e-12))

        log_cdf_delta = torch.where(
            x_rp < 1e-3,
            log_cdf_0,
            torch.where(
                x_rp > self.d - 1 - 1e-3,
                log_cdf_d_1,
                torch.log(torch.clamp(cdf_delta, min=1e-12)),
            ),
        )
        log_pi = F.log_softmax(self.pi, dim=0)
        log_probs = log_cdf_delta + log_pi
        log_probs = torch.logsumexp(log_probs, dim=1)

        return log_probs

    def loss(self, x):
        return -torch.mean(self(x))

    def get_distribution(self):
        with torch.no_grad():
            x = torch.FloatTensor(np.arange(self.d)).cuda()
            distribution = self(x).exp()
        return distribution.detach().cpu().numpy()
