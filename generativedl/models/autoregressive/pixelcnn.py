class MaskedConv2d(nn.Conv2d):
    def __init__(
            self,
            mask_type,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            **kwargs,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            **kwargs,
        )
        self.mask_type = mask_type
        self.register_buffer("mask", torch.zeros_like(self.weight))
        k = self.kernel_size[0]
        self.mask[:, :, : k // 2] = 1
        self.mask[:, :, k // 2, : k // 2] = 1
        if self.mask_type == "B":
            self.mask[:, :, k // 2, k // 2] = 1

    def forward(self, input):
        # mask * weight to prevent certain weights from contributing / being updated
        return F.conv2d(
            input,
            self.weight * self.mask,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )


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


class ResBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            kernel_size=7,
            **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels

        self.block = [
            MaskedConv2d(
                mask_type="B",
                in_channels=in_channels,
                out_channels=in_channels // 2,
                kernel_size=1,
                padding=0,
                **kwargs,
            ),
            nn.ReLU(),
            MaskedConv2d(
                mask_type="B",
                in_channels=in_channels // 2,
                out_channels=in_channels // 2,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                **kwargs,
            ),
            nn.ReLU(),
            MaskedConv2d(
                mask_type="B",
                in_channels=in_channels // 2,
                out_channels=in_channels,
                kernel_size=1,
                padding=0,
                **kwargs,
            ),
            nn.ReLU(),
        ]
        self.block = nn.Sequential(*self.block)

        return

    def forward(self, x):
        return self.block(x) + x


class PixelCNN(nn.Module):
    def __init__(
            self,
            input_shape,
            n_colors,
            kernel_size=7,
            n_filters=64,
            n_layers=5,
            **kwargs,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.n_filters = n_filters
        self.n_layers = n_layers
        self.n_colors = n_colors
        self.n_channels = n_channels = input_shape[0]

        self.net = [
            MaskedConv2d(
                mask_type="A",
                in_channels=n_channels,
                out_channels=n_filters,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                **kwargs,
            ),
            nn.ReLU(),
        ]
        for _ in range(n_layers):
            self.net.extend(
                [
                    MaskedConv2d(
                        mask_type="B",
                        in_channels=n_filters,
                        out_channels=n_filters,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                        **kwargs,
                    ),
                    nn.ReLU(),
                ]
            )
        self.net.extend(
            [
                MaskedConv2d(
                    mask_type="B",
                    in_channels=n_filters,
                    out_channels=n_filters,
                    kernel_size=1,
                    padding=0,
                    **kwargs,
                ),
                nn.ReLU(),
                MaskedConv2d(
                    mask_type="B",
                    in_channels=n_filters,
                    out_channels=n_channels * n_colors,
                    kernel_size=1,
                    padding=0,
                    **kwargs,
                ),
            ]
        )
        self.net = nn.Sequential(*self.net)

        return

    def forward(self, x):
        out = x.float() / (self.n_colors - 1) - 0.5
        out = self.net(out)

        return out.view(x.shape[0], self.n_colors, *self.input_shape)

    def loss(self, x):
        out = self(x)
        loss = F.cross_entropy(out, x.long())

        return loss

    def sample(
            self,
            n,
    ):
        samples = torch.zeros(n, *self.input_shape).cuda()
        with torch.no_grad():
            for r in range(self.input_shape[1]):
                for c in range(self.input_shape[2]):
                    for k in range(self.n_channels):
                        logits = self(samples)[:, :, k, r, c]
                        probs = F.softmax(logits, dim=1)
                        samples[:, k, r, c] = torch.multinomial(
                            probs, 1
                        ).squeeze(-1)
        return samples.permute(0, 2, 3, 1).cpu().numpy()
