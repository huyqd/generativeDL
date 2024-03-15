import torch
import torch.nn.functional as F
from models.masked import (
    GatedMaskedConv,
    VerticalMaskedConvolution,
    HorizontalMaskedConvolution,
    MaskedConvolution,
)
from torch import nn


class PixelCNN(nn.Module):
    def __init__(self, c_in, c_hidden):
        super().__init__()
        # Initial convolutions skipping the center pixel
        # Convolution block of PixelCNN. We use dilation instead of downscaling
        self.conv_layers = nn.Sequential(
            *[
                MaskedConvolution(c_in, c_hidden, kernel_size=7, mask_center=True),
                nn.ReLU(),
                MaskedConvolution(c_hidden, c_hidden, kernel_size=7),
                nn.ReLU(),
                MaskedConvolution(c_hidden, c_hidden, kernel_size=7),
                nn.ReLU(),
                MaskedConvolution(c_hidden, c_hidden, kernel_size=7),
                nn.ReLU(),
                MaskedConvolution(c_hidden, c_hidden, kernel_size=7),
                nn.ReLU(),
                MaskedConvolution(c_hidden, c_hidden, kernel_size=7),
                nn.ReLU(),
                MaskedConvolution(c_hidden, c_hidden, kernel_size=1),
                nn.ReLU(),
                MaskedConvolution(c_hidden, c_in * 256, kernel_size=1),
            ]
        )

    def forward(self, x):
        """Forward image through model and return logits for each pixel.

        Args:
            x: Image tensor with integer values between 0 and 255.
        """
        # Scale input from 0 to 255 back to -1 to 1
        x = (x.float() / 255.0) * 2 - 1
        # x = (x.float() - 0.5) / 0.5

        out = self.conv_layers(x)

        # Output dimensions: [Batch, Classes, Channels, Height, Width]
        out = out.reshape(out.shape[0], 256, out.shape[1] // 256, out.shape[2], out.shape[3])
        # out = out.reshape(out.shape[0], 2, out.shape[1] // 2, out.shape[2], out.shape[3])

        return out


class GatedPixelCNN(nn.Module):
    def __init__(self, c_in, c_hidden):
        super().__init__()
        # Initial convolutions skipping the center pixel
        self.conv_vstack = VerticalMaskedConvolution(c_in, c_hidden, mask_center=True)
        self.conv_hstack = HorizontalMaskedConvolution(c_in, c_hidden, mask_center=True)
        # Convolution block of PixelCNN. We use dilation instead of downscaling
        self.conv_layers = nn.ModuleList(
            [
                GatedMaskedConv(c_hidden),
                GatedMaskedConv(c_hidden, dilation=2),
                GatedMaskedConv(c_hidden),
                GatedMaskedConv(c_hidden, dilation=4),
                GatedMaskedConv(c_hidden),
                GatedMaskedConv(c_hidden, dilation=2),
                GatedMaskedConv(c_hidden),
            ]
        )
        # Output classification convolution (1x1)
        self.conv_out = nn.Conv2d(c_hidden, c_in * 256, kernel_size=1, padding=0)
        # self.conv_out = nn.Conv2d(c_hidden, c_in * 2, kernel_size=1, padding=0)

    def forward(self, x):
        """Forward image through model and return logits for each pixel.

        Args:
            x: Image tensor with integer values between 0 and 255.
        """
        # Scale input from 0 to 255 back to -1 to 1
        x = (x.float() / 255.0) * 2 - 1
        # x = (x.float() - 0.5) / 0.5

        # Initial convolutions
        v_stack = self.conv_vstack(x)
        h_stack = self.conv_hstack(x)
        # Gated Convolutions
        for layer in self.conv_layers:
            v_stack, h_stack = layer(v_stack, h_stack)
        # 1x1 classification convolution
        # Apply ELU before 1x1 convolution for non-linearity on residual connection
        out = self.conv_out(F.elu(h_stack))

        # Output dimensions: [Batch, Classes, Channels, Height, Width]
        out = out.reshape(out.shape[0], 256, out.shape[1] // 256, out.shape[2], out.shape[3])
        # out = out.reshape(out.shape[0], 2, out.shape[1] // 2, out.shape[2], out.shape[3])

        return out


class MyMaskConv2d(nn.Module):
    def __init__(self, mask_center, *args, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(*args, **kwargs)
        self.register_buffer("mask", torch.ones(self.conv.kernel_size[0], self.conv.kernel_size[0]))
        self.create_mask(mask_center)

    def forward(self, input):
        self.conv.weight.data *= self.mask
        return self.conv(input)

    def create_mask(self, mask_center):
        kernel_size = self.conv.kernel_size[0]
        self.mask[(kernel_size // 2 + 1) :, :] = 0
        self.mask[kernel_size // 2, kernel_size // 2 + 1 :] = 0

        if mask_center:
            self.mask[kernel_size // 2, kernel_size // 2] = 0

        self.mask = self.mask[None, None]


class MyPixelCNN(nn.Module):
    def __init__(
        self,
        c_in,
        c_hidden,
        n_colors=256,
        n_filters=64,
        kernel_size=7,
        n_layers=5,
        **kwargs,
    ):
        super().__init__()
        assert n_layers >= 2

        input_shape = (1, 28, 28)
        n_channels = input_shape[0]
        model = nn.Sequential(
            *[
                MyMaskConv2d(True, n_channels, n_filters, kernel_size=kernel_size, padding=kernel_size // 2, **kwargs),
                nn.ReLU(),
                MyMaskConv2d(False, n_filters, n_filters, kernel_size=kernel_size, padding=kernel_size // 2, **kwargs),
                nn.ReLU(),
                MyMaskConv2d(False, n_filters, n_filters, kernel_size=kernel_size, padding=kernel_size // 2, **kwargs),
                nn.ReLU(),
                MyMaskConv2d(False, n_filters, n_filters, kernel_size=kernel_size, padding=kernel_size // 2, **kwargs),
                nn.ReLU(),
                MyMaskConv2d(False, n_filters, n_filters, kernel_size=kernel_size, padding=kernel_size // 2, **kwargs),
                nn.ReLU(),
                MyMaskConv2d(False, n_filters, n_filters, kernel_size=kernel_size, padding=kernel_size // 2, **kwargs),
                nn.ReLU(),
                MyMaskConv2d(False, n_filters, n_filters, kernel_size=1, **kwargs),
                nn.ReLU(),
                MyMaskConv2d(False, n_filters, n_colors * n_channels, kernel_size=1, **kwargs),
            ]
        )

        self.net = model
        self.input_shape = input_shape
        self.n_colors = n_colors
        self.n_channels = n_channels

    def forward(self, x):
        batch_size = x.shape[0]
        out = (x.float() / 255.0) * 2 - 1
        # out = (x.float() / -0.5) / 0.5
        out = self.net(out)

        return out.view(batch_size, self.n_colors, *self.input_shape)

    def loss(self, x, cond=None):
        return F.cross_entropy(self(x, cond=cond), x.long())

    @torch.no_grad()
    def sample(self, n):
        # samples = torch.zeros(n, *self.input_shape).cuda()
        samples = torch.zeros(n, *self.input_shape)
        for h in range(self.input_shape[1]):
            for w in range(self.input_shape[2]):
                for c in range(self.n_channels):
                    logits = self(samples)[:, :, c, h, w]
                    probs = F.softmax(logits, dim=1)
                    samples[:, c, h, w] = torch.multinomial(probs, 1).squeeze(-1)
        return samples.permute(0, 2, 3, 1).cpu().numpy()
