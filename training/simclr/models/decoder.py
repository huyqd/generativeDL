from torch import nn


class Decoder(nn.Module):
    def __init__(self, c_in: int, c_out: int, latent_dim: int, act_fn: callable = nn.GELU):
        """Decoder.

        Args:
           c_in : Number of channels of the image to reconstruct. For CIFAR, this parameter is 3
           c_out : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
           latent_dim : Dimensionality of latent representation z
           act_fn : Activation function used throughout the decoder network
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 2 * 16 * c_out),
            act_fn(),
            nn.Unflatten(1, (2 * c_out, 4, 4)),
            nn.ConvTranspose2d(2 * c_out, 2 * c_out, kernel_size=3, output_padding=1, padding=1, stride=2),
            # 4x4 => 8x8
            act_fn(),
            nn.Conv2d(2 * c_out, 2 * c_out, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(2 * c_out, c_out, kernel_size=3, output_padding=1, padding=1, stride=2),  # 8x8 => 16x16
            act_fn(),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(c_out, c_in, kernel_size=3, output_padding=1, padding=1, stride=2),  # 16x16 => 32x32
            nn.Tanh(),  # The input images is scaled between -1 and 1, hence the output has to be bounded as well
        )

    def forward(self, x):
        x = self.net(x)
        return x
