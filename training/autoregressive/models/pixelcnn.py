import copy
from collections import OrderedDict

import torch.nn.functional as F
from models.masked import (
    MaskedConv2d,
    GatedMaskedConv,
    VerticalMaskedConv2d,
    HorizontalMaskedConv2d,
)
from models.residual import ResidualBlock
from torch import nn


class GatedPixelCNN(nn.Module):
    def __init__(self, c_in, c_hidden):
        super().__init__()
        # Initial convolutions skipping the center pixel
        self.conv_vstack = VerticalMaskedConv2d(c_in, c_hidden, mask_center=True)
        self.conv_hstack = HorizontalMaskedConv2d(c_in, c_hidden, mask_center=True)
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

        return out


class LayerNorm(nn.LayerNorm):
    def __init__(
        self,
        in_shape,
        **kwargs,
    ):
        super().__init__(in_shape, **kwargs)

        return

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()

        return super().forward(x).permute(0, 3, 1, 2).contiguous()


class PixelCNN(nn.Module):
    def __init__(
        self,
        input_shape,
        n_colors,
        n_filters=64,
        kernel_size=7,
        n_layers=5,
        use_resblock: bool = False,
        **kwargs,
    ):
        super().__init__()
        assert n_layers >= 2

        self.input_shape = input_shape
        self.n_colors = n_colors
        self.n_channels = input_shape[0]

        initial_masked_conv = MaskedConv2d(
            True,
            in_channels=self.n_channels,
            out_channels=n_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            **kwargs,
        )

        if use_resblock:
            mid_masked_conv = ResidualBlock(n_filters)
        else:
            mid_masked_conv = MaskedConv2d(
                False,
                in_channels=n_filters,
                out_channels=n_filters,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                **kwargs,
            )

        down_masked_conv = MaskedConv2d(
            False,
            in_channels=n_filters,
            out_channels=n_filters,
            kernel_size=1,
            **kwargs,
        )
        final_masked_conv = MaskedConv2d(
            False,
            in_channels=n_filters,
            out_channels=self.n_colors * self.n_channels,
            kernel_size=1,
            **kwargs,
        )

        layers = OrderedDict(
            [
                ("inital_masked_conv", initial_masked_conv),
                ("initial_layer_norm", LayerNorm(n_filters)),
                ("initial_relu", nn.ReLU()),
            ]
        )

        for l in range(n_layers):
            if use_resblock:
                layers.update(
                    [
                        (f"mid_residualblock{l}", copy.deepcopy(mid_masked_conv)),
                        (f"mid_layer_norm{l}", LayerNorm(n_filters)),
                    ]
                )
            else:
                layers.update(
                    [
                        (f"mid_masked_conv{l}", copy.deepcopy(mid_masked_conv)),
                        (f"middle_relu{l}", nn.ReLU()),
                    ]
                )

        layers.update(
            [
                ("down_masked_conv", down_masked_conv),
                ("down_relu", nn.ReLU()),
                ("final_masked_conv", final_masked_conv),
            ]
        )

        self.net = nn.Sequential(layers)

    def forward(self, x):
        batch_size = x.shape[0]
        out = (x.float() / (self.n_colors - 1)) * 2 - 1
        out = self.net(out)

        return out.view(batch_size, self.n_colors, *self.input_shape)
