from types import SimpleNamespace

from models.activation_functions import ACT_FN_BY_NAME
from torch import nn


class ResNetBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, act_fn, downsample: bool):
        """ResNetBlock.

        Args:
            c_in: Number of input features
            c_out - Number of output features. Note that this is only relevant if downsample is True, as otherwise, c_out = c_in
            act_fn: Activation class constructor (e.g. nn.ReLU)
            downsample - If True, we want to apply a stride inside the block and reduce the output shape by 2 in height and width

        """
        super().__init__()
        c_out = c_in if not downsample else c_out
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, stride=2 if downsample else 1, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            act_fn(),
            nn.Conv2d(c_out, c_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
        )
        self.downsample = nn.Conv2d(c_in, c_out, kernel_size=1, stride=2, padding=0) if downsample else None
        self.act_fn = act_fn()

    def forward(self, x):
        z = self.net(x)
        if self.downsample is not None:
            x = self.downsample(x)
        out = self.act_fn(z + x)

        return out


class PreActResNetBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, act_fn, downsample: bool):
        """PreAct ResNetBlock.

        Args:
            c_in: Number of input features
            c_out - Number of output features. Note that this is only relevant if downsample is True, as otherwise, c_out = c_in
            act_fn: Activation class constructor (e.g. nn.ReLU)
            downsample - If True, we want to apply a stride inside the block and reduce the output shape by 2 in height and width

        """
        super().__init__()
        c_out = c_in if not downsample else c_out
        self.net = nn.Sequential(
            nn.BatchNorm2d(c_in),
            act_fn(),
            nn.Conv2d(c_in, c_out, kernel_size=3, stride=2 if downsample else 1, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            act_fn(),
            nn.Conv2d(c_out, c_out, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.downsample = (
            nn.Sequential(nn.BatchNorm2d(c_in), act_fn(), nn.Conv2d(c_in, c_out, kernel_size=1, stride=2, padding=0))
            if downsample
            else None,
        )

    def forward(self, x):
        z = self.net(x)
        if self.downsample is not None:
            x = self.downsample(x)

        return x + z


resnet_blocks_name = {"ResNetBlock": ResNetBlock, "PreActResNetBlock": PreActResNetBlock}


class ResNet(nn.Module):
    def __init__(
        self,
        num_classes=10,
        num_blocks=(3, 3, 3),
        c_hidden=(16, 32, 64),
        act_fn_name="relu",
        block_name="ResNetBlock",
        **kwargs,
    ):
        super().__init__()
        self.hparams = SimpleNamespace(
            num_classes=num_classes,
            c_hidden=c_hidden,
            num_blocks=num_blocks,
            act_fn_name=act_fn_name,
            act_fn=ACT_FN_BY_NAME[act_fn_name],
            block_class=resnet_blocks_name[block_name],
        )
        self._create_network()
        self._init_params()

    def _create_network(self):
        c_hidden = self.hparams.c_hidden
        act_fn = self.hparams.act_fn

        # A first convolution on the original image to scale up the channel size
        self.input_net = nn.Sequential(
            nn.Conv2d(3, c_hidden[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_hidden[0]),
            act_fn(),
        )

        resnet_blocks = []
        for block_idx, num_blocks in enumerate(self.hparams.num_blocks):
            for i in range(num_blocks):
                downsample = i == 0 and block_idx > 0
                resnet_blocks.append(
                    self.hparams.block_class(
                        c_hidden[block_idx if not downsample else block_idx - 1],
                        c_hidden[block_idx],
                        act_fn,
                        downsample,
                    )
                )

        self.resnet_blocks = nn.Sequential(*resnet_blocks)

        # Mapping to classification output
        self.output_net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(c_hidden[-1], self.hparams.num_classes),
        )

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Fan-out focuses on the gradient distribution, and is commonly used in ResNets
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity=self.hparams.act_fn_name)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.input_net(x)
        x = self.resnet_blocks(x)
        x = self.output_net(x)

        return x
