import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def to_one_hot(labels, d):
    """Scatter 1 at ith column of label i, matching row of labels"""
    # d: The number of possible discrete values for x_i
    # Note to self: one hot encode input to turn categorical(ish) data into numeric,
    # otherwise we inadvertently encode ordering meaning with the input -> worsen training since the loss is (categorial) entropy loss
    one_hot = torch.zeros(labels.shape[0], d).cuda()
    one_hot.scatter_(1, labels.unsqueeze(1), 1)
    return one_hot


# Code based one Andrej Karpathy's implementation: https://github.com/karpathy/pytorch-made
class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        # Create weights and bias but will not be updated with backprop
        self.register_buffer("mask", torch.ones(out_features, in_features))

    def set_mask(self, mask):
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T))

    def forward(self, input):
        # mask * weight to prevent certain weights from contributing / being updated
        return F.linear(input, self.mask * self.weight, self.bias)


class MADE(nn.Module):
    def __init__(
        self,
        input_shape,
        d,
        hidden_size=[512, 512, 512],
        ordering=None,
        one_hot_input=False,
    ):
        # d: The number of possible discrete values for x_i
        super().__init__()
        self.input_shape = input_shape
        self.nin = np.prod(input_shape)
        self.nout = self.nin * d  # if input is 2 pixel -> output d (possible values for 1 pixel) * 2
        self.d = d
        self.hidden_sizes = hidden_size
        self.ordering = np.arange(self.nin) if ordering is None else ordering
        self.one_hot_input = one_hot_input

        # define a simple MLP neural net
        self.net = []
        hs = [self.nin * d if one_hot_input else self.nin] + self.hidden_sizes + [self.nout]
        for h0, h1 in zip(hs, hs[1:]):
            self.net.extend(
                [
                    MaskedLinear(h0, h1),
                    nn.ReLU(),
                ]
            )
        self.net.pop()  # pop the last ReLU for the output layer
        self.net = nn.Sequential(*self.net)

        self.m = {}
        self.create_mask()  # builds the initial self.m connectivity

    def create_mask(self):
        L = len(self.hidden_sizes)

        # sample the order of the inputs and the connectivity of all neurons
        self.m[-1] = self.ordering
        for l in range(L):
            self.m[l] = np.random.randint(self.m[l - 1].min(), self.nin - 1, size=self.hidden_sizes[l])

        # construct the mask matrices, or rather unmask, where True is a connection, while False means no connection
        masks = [self.m[l - 1][:, None] <= self.m[l][None, :] for l in range(L)]
        masks.append(self.m[L - 1][:, None] < self.m[-1][None, :])

        masks[-1] = np.repeat(masks[-1], self.d, axis=1)
        if self.one_hot_input:
            masks[0] = np.repeat(masks[0], self.d, axis=0)

        # set the masks in all MaskedLinear layers
        layers = [l for l in self.net.modules() if isinstance(l, MaskedLinear)]
        for l, m in zip(layers, masks):
            l.set_mask(m)

    def forward(self, x):
        batch_size = x.shape[0]
        if self.one_hot_input:
            x = x.long().contiguous().view(-1)
            x = to_one_hot(x, self.d)
            x = x.view(batch_size, -1)
        else:
            x = x.float()
            x = x.view(batch_size, self.nin)
        logits = self.net(x).view(batch_size, self.nin, self.d)
        return (
            logits.permute(0, 2, 1)
            .contiguous()
            .view(batch_size, self.d, *self.input_shape)  # refer to this page to see why we need to reshape this way
            # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        )

    def loss(self, x):
        return F.cross_entropy(self(x), x.long())

    def sample(self, n):
        samples = torch.zeros(n, self.nin)
        self.inv_ordering = {x: i for i, x in enumerate(self.ordering)}
        with torch.no_grad():
            for i in range(self.nin):
                logits = self(samples).view(n, self.d, self.nin)[:, :, self.inv_ordering[i]]
                probs = F.softmax(logits, dim=1)
                samples[:, self.inv_ordering[i]] = torch.multinomial(probs, 1).squeeze(-1)
            samples = samples.view(n, *self.input_shape)
        return samples.cpu().numpy()

    def get_distribution(self):
        assert self.input_shape == (2,), "Only available for 2D joint"
        x = np.mgrid[0 : self.d, 0 : self.d].reshape(2, self.d**2).T
        x = torch.LongTensor(x).cuda()
        log_probs = F.log_softmax(self(x), dim=1)
        distribution = torch.gather(log_probs, 1, x.unsqueeze(1)).squeeze(1)
        distribution = distribution.sum(dim=1)

        return distribution.exp().view(self.d, self.d).detach().cpu().numpy()
