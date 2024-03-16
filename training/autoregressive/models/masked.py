import torch
import torch.nn.functional as F
from torch import nn


class BaseMaskedConv2d(nn.Conv2d):
    def __init__(self, mask_center=None, *args, **kwargs):
        """Implements a convolution with mask applied on its weights.

        Args:
            c_in: Number of input channels
            c_out: Number of output channels
            mask: Tensor of shape [kernel_size_H, kernel_size_W] with 0s where
                   the convolution should be masked, and 1s otherwise.
            kwargs: Additional arguments for the convolution
        """
        super().__init__(*args, **kwargs)
        # For simplicity: calculate padding automatically
        self.register_buffer("mask", torch.ones(self.kernel_size[0], self.kernel_size[0]))
        self.create_mask(mask_center)

    def create_mask(self, mask_center):
        kernel_size = self.kernel_size[0]
        self.mask[(kernel_size // 2 + 1) :, :] = 0
        self.mask[kernel_size // 2, kernel_size // 2 + 1 :] = 0

        if mask_center:
            self.mask[kernel_size // 2, kernel_size // 2] = 0

        self.mask = self.mask[None, None]

    def forward(self, x):
        return F.conv2d(
            x,
            self.weight * self.mask,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )


class MaskedConv2d(BaseMaskedConv2d):
    def __init__(self, mask_center, *args, **kwargs):
        super().__init__(mask_center=mask_center, *args, **kwargs)


class VerticalMaskedConv2d(BaseMaskedConv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, mask_center=False, **kwargs):
        # Mask out all pixels below. For efficiency, we could also reduce the kernel
        # size in height, but for simplicity, we stick with masking here.
        mask = torch.ones(kernel_size, kernel_size)
        mask[(kernel_size // 2 + 1) :, :] = 0

        # For the very first convolution, we will also mask the center row
        if mask_center:
            mask[kernel_size // 2, :] = 0

        super().__init__(
            mask_center,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            **kwargs,
        )
        self.mask = mask


class HorizontalMaskedConv2d(BaseMaskedConv2d):
    def __init__(self, c_in, c_out, kernel_size=3, mask_center=False, **kwargs):
        # Mask out all pixels on the left. Note that our kernel has a size of 1
        # in height because we only look at the pixel in the same row.
        mask = torch.ones(1, kernel_size)
        mask[0, (kernel_size // 2 + 1) :] = 0

        # For the very first convolution, we will also mask the center pixel
        if mask_center:
            mask[0, kernel_size // 2] = 0

        super().__init__(
            mask_center,
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=kernel_size,
            **kwargs,
        )
        self.mask = mask


class GatedMaskedConv(nn.Module):
    """
    https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial12/Autoregressive_Image_Modeling.html#Gated-Convolutions
    """

    def __init__(self, c_in, **kwargs):
        """Gated Convolution block implemented the computation graph shown above."""
        super().__init__()
        self.conv_vert = VerticalMaskedConv2d(c_in, out_channels=2 * c_in, **kwargs)
        self.conv_horiz = HorizontalMaskedConv2d(c_in, c_out=2 * c_in, **kwargs)
        self.conv_vert_to_horiz = nn.Conv2d(2 * c_in, 2 * c_in, kernel_size=1, padding=0)
        self.conv_horiz_1x1 = nn.Conv2d(c_in, c_in, kernel_size=1, padding=0)

    def forward(self, v_stack, h_stack):
        # Vertical stack (left)
        v_stack_feat = self.conv_vert(v_stack)
        v_val, v_gate = v_stack_feat.chunk(2, dim=1)
        v_stack_out = torch.tanh(v_val) * torch.sigmoid(v_gate)

        # Horizontal stack (right)
        h_stack_feat = self.conv_horiz(h_stack)
        h_stack_feat = h_stack_feat + self.conv_vert_to_horiz(v_stack_feat)
        h_val, h_gate = h_stack_feat.chunk(2, dim=1)
        h_stack_feat = torch.tanh(h_val) * torch.sigmoid(h_gate)
        h_stack_out = self.conv_horiz_1x1(h_stack_feat)
        h_stack_out = h_stack_out + h_stack

        return v_stack_out, h_stack_out
