import torch.nn.functional as F
from torch import nn

from models.masked import (
    GatedMaskedConv,
    VerticalMaskedConvolution,
    HorizontalMaskedConvolution,
    MaskedConvolution,
)


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
