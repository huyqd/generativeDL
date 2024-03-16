from models.masked import MaskedConv2d
from torch import nn


class ResidualBlock(nn.Module):
    """
    One residual layer inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    """

    def __init__(self, n_filters):
        super().__init__()
        self.res_block = nn.Sequential(
            MaskedConv2d(
                False,
                in_channels=n_filters,
                out_channels=n_filters,
                kernel_size=1,
            ),
            nn.ReLU(),
            MaskedConv2d(
                False,
                in_channels=n_filters,
                out_channels=n_filters,
                kernel_size=7,
                padding=7 // 2,
            ),
            nn.ReLU(),
            MaskedConv2d(
                False,
                in_channels=n_filters,
                out_channels=n_filters,
                kernel_size=1,
            ),
            nn.ReLU(),
        )

    def forward(self, x):
        x = x + self.res_block(x)
        return x
