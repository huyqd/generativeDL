import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from tqdm import tqdm

from training.autoregressive.models.masked import VerticalStackConvolution, HorizontalStackConvolution

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    ACCELERATOR = "gpu"
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    ACCELERATOR = "mps"
else:
    DEVICE = torch.device("cpu")
    ACCELERATOR = "cpu"


class GatedMaskedConv(nn.Module):
    def __init__(self, c_in, **kwargs):
        """Gated Convolution block implemented the computation graph shown above."""
        super().__init__()
        self.conv_vert = VerticalStackConvolution(c_in, c_out=2 * c_in, **kwargs)
        self.conv_horiz = HorizontalStackConvolution(c_in, c_out=2 * c_in, **kwargs)
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


class PixelCNN(L.LightningModule):
    def __init__(self, c_in, c_hidden):
        super().__init__()
        self.save_hyperparameters()

        # Initial convolutions skipping the center pixel
        self.conv_vstack = VerticalStackConvolution(c_in, c_hidden, mask_center=True)
        self.conv_hstack = HorizontalStackConvolution(c_in, c_hidden, mask_center=True)
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

        self.example_input_array = torch.randn(1, 1, 28, 28)

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

    def calc_likelihood(self, x):
        # Forward pass with bpd likelihood calculation
        pred = self.forward(x)
        nll = F.cross_entropy(pred, x, reduction="none")
        bpd = nll.mean(dim=[1, 2, 3]) * np.log2(np.exp(1))
        return bpd.mean()

    @torch.no_grad()
    def sample(self, img_shape, img=None):
        """Sampling function for the autoregressive model.

        Args:
            img_shape: Shape of the image to generate (B,C,H,W)
            img (optional): If given, this tensor will be used as
                             a starting image. The pixels to fill
                             should be -1 in the input tensor.
        """
        # Create empty image
        if img is None:
            img = torch.zeros(img_shape, dtype=torch.long).to(DEVICE) - 1
        # Generation loop
        for h in tqdm(range(img_shape[2]), leave=False):
            for w in range(img_shape[3]):
                for c in range(img_shape[1]):
                    # Skip if not to be filled (-1)
                    if (img[:, c, h, w] != -1).all().item():
                        continue
                    # For efficiency, we only have to input the upper part of the image
                    # as all other parts will be skipped by the masked convolutions anyways
                    pred = self.forward(img[:, :, : h + 1, :])
                    probs = F.softmax(pred[:, :, c, h, w], dim=-1)
                    img[:, c, h, w] = torch.multinomial(probs, num_samples=1).squeeze(dim=-1)
        return img

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        loss = self.calc_likelihood(batch[0])
        self.log("train_bpd", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.calc_likelihood(batch[0])
        self.log("val_bpd", loss)

    def test_step(self, batch, batch_idx):
        loss = self.calc_likelihood(batch[0])
        self.log("test_bpd", loss)
