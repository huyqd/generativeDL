import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch import Tensor

from training.autoregressive.models.masked import (
    HorizontalMaskedConvolution,
    VerticalMaskedConvolution,
    MaskedConvolution,
)


def show_imgs(imgs):
    num_imgs = imgs.shape[0] if isinstance(imgs, Tensor) else len(imgs)
    nrow = min(num_imgs, 4)
    ncol = int(math.ceil(num_imgs / nrow))
    imgs = torchvision.utils.make_grid(imgs, nrow=nrow, pad_value=128)
    imgs = imgs.clamp(min=0, max=255)
    np_imgs = imgs.cpu().numpy()
    plt.figure(figsize=(1.5 * nrow, 1.5 * ncol))
    plt.imshow(np.transpose(np_imgs, (1, 2, 0)), interpolation="nearest")
    plt.axis("off")
    plt.show()
    plt.close()


def show_center_recep_field(img, out):
    """Calculates the gradients of the input with respect to the output center pixel, and visualizes the overall
    receptive field.

    Args:
        img: Input image for which we want to calculate the receptive field on.
        out: Output features/loss which is used for backpropagation, and should be
              the output of the network/computation graph.
    """
    # Determine gradients
    loss = out[0, :, img.shape[2] // 2, img.shape[3] // 2].sum()  # L1 loss for simplicity
    # Retain graph as we want to stack multiple layers and show the receptive field of all of them
    loss.backward(retain_graph=True)
    img_grads = img.grad.abs()
    img.grad.fill_(0)  # Reset grads

    # Plot receptive field
    img = img_grads.squeeze().cpu().numpy()
    fig, ax = plt.subplots(1, 2)
    _ = ax[0].imshow(img)
    ax[1].imshow(img > 0)
    # Mark the center pixel in red if it doesn't have any gradients (should be
    # the case for standard autoregressive models)
    show_center = img[img.shape[0] // 2, img.shape[1] // 2] == 0
    if show_center:
        center_pixel = np.zeros(img.shape + (4,))
        center_pixel[center_pixel.shape[0] // 2, center_pixel.shape[1] // 2, :] = np.array([1.0, 0.0, 0.0, 1.0])

        for i in range(2):
            ax[i].axis("off")
            if show_center:
                ax[i].imshow(center_pixel)

    ax[0].set_title("Weighted receptive field")
    ax[1].set_title("Binary receptive field")
    plt.show()
    plt.close()


if __name__ == "__main__":
    # %%
    inp_img = torch.ones(1, 1, 11, 11)
    inp_img.requires_grad_()
    show_center_recep_field(inp_img, inp_img)

    # %% Normal MaskedConvolution
    masked_conv = MaskedConvolution(c_in=1, c_out=1, kernel_size=3, mask_center=True)
    subsequent_masked_conv = MaskedConvolution(c_in=1, c_out=1, kernel_size=3, mask_center=False)
    masked_conv.conv.weight.data.fill_(1)
    masked_conv.conv.bias.data.fill_(0)
    subsequent_masked_conv.conv.weight.data.fill_(1)
    subsequent_masked_conv.conv.bias.data.fill_(0)
    masked_img = masked_conv(inp_img)
    show_center_recep_field(inp_img, masked_img)

    # %% multiple StackMaskedConvolution
    for l_idx in range(4):
        masked_img = subsequent_masked_conv(masked_img)
        print("Layer %i" % (l_idx + 2))
        show_center_recep_field(inp_img, masked_img)

    # %% HorizontalStackConvolution
    horiz_conv = HorizontalMaskedConvolution(c_in=1, c_out=1, kernel_size=3, mask_center=True)
    horiz_conv.conv.weight.data.fill_(1)
    horiz_conv.conv.bias.data.fill_(0)
    horiz_img = horiz_conv(inp_img)
    show_center_recep_field(inp_img, horiz_img)

    # %% VerticalStackConvolution
    vert_conv = VerticalMaskedConvolution(c_in=1, c_out=1, kernel_size=3, mask_center=True)
    vert_conv.conv.weight.data.fill_(1)
    vert_conv.conv.bias.data.fill_(0)
    vert_img = vert_conv(inp_img)
    show_center_recep_field(inp_img, vert_img)

    # %% comebine
    horiz_img = vert_img + horiz_img
    show_center_recep_field(inp_img, horiz_img)

    # %% layer
    # Initialize convolutions with equal weight to all input pixels
    horiz_conv = HorizontalMaskedConvolution(c_in=1, c_out=1, kernel_size=3, mask_center=False)
    horiz_conv.conv.weight.data.fill_(1)
    horiz_conv.conv.bias.data.fill_(0)
    vert_conv = VerticalMaskedConvolution(c_in=1, c_out=1, kernel_size=3, mask_center=False)
    vert_conv.conv.weight.data.fill_(1)
    vert_conv.conv.bias.data.fill_(0)

    # We reuse our convolutions for the 4 layers here. Note that in a standard network,
    # we don't do that, and instead learn 4 separate convolution. As this cell is only for
    # visualization purposes, we reuse the convolutions for all layers.
    for l_idx in range(4):
        vert_img = vert_conv(vert_img)
        horiz_img = horiz_conv(horiz_img) + vert_img
        print("Layer %i" % (l_idx + 2))
        show_center_recep_field(inp_img, horiz_img)
