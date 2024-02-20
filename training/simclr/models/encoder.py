from torch import nn


class Encoder(nn.Module):
    def __init__(self, c_in: int, c_out: int, latent_dim: int, act_fn: callable = nn.GELU):
        """Encoder.

        Args:
           c_in : Number of input channels of the image. For CIFAR, this parameter is 3
           c_out : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
           latent_dim : Dimensionality of latent representation z
           act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, stride=2),  # 32x32 => 16x16
            act_fn(),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(c_out, 2 * c_out, kernel_size=3, padding=1, stride=2),  # 16x16 => 8x8
            act_fn(),
            nn.Conv2d(2 * c_out, 2 * c_out, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(2 * c_out, 2 * c_out, kernel_size=3, padding=1, stride=2),  # 8x8 => 4x4
            act_fn(),
            nn.Flatten(),  # Image grid to single feature vector
            nn.Linear(2 * 16 * c_out, latent_dim),
        )

    def forward(self, x):
        return self.net(x)
