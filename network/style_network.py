import torch
import torch.nn as nn

import config

img_channels = config.IMG_CHANNELS
down_channels = config.DOWN_CHANNELS
num_res = config.NUM_RES
up_channels = config.UP_CHANNELS
final_channels = config.FINAL_CHANNELS


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        kernel_size=3,
        padding=1,
        output_padding=1,
        up_sampling: bool = False,
        activation: nn.Module = None,
    ):
        super().__init__()
        
        self.conv = (
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                padding_mode="reflect",
            )
            if not up_sampling
            else nn.Sequential(
                nn.Upsample(scale_factor=stride, mode="bilinear"),
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    padding=padding,
                    padding_mode="reflect"
                )
            )
        )
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        self.activation = activation if activation else nn.Identity()

    def forward(self, x):
        x = self.norm(self.conv(x))
        return self.activation(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv1 = ConvBlock(
            in_channels,
            in_channels,
            1,
            kernel_size,
            padding,
            activation=nn.ReLU(),
        )
        self.conv2 = ConvBlock(
            in_channels, in_channels, 1, kernel_size, padding
        )

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(x)


class StylizingNetwork(nn.Module):
    def __init__(
        self,
        img_channels=img_channels,
        down_channels=down_channels,
        num_res=num_res,
        up_channels=up_channels,
        final_channels=final_channels,
    ):
        super().__init__()

        self.initial_block = ConvBlock(
            img_channels, down_channels[0], 1, activation=nn.ReLU()
        )

        in_channels = down_channels[0]
        self.down_blocks = nn.ModuleList()
        for out_channels in down_channels[1:]:
            self.down_blocks.append(
                ConvBlock(in_channels, out_channels, 2, activation=nn.ReLU())
            )
            in_channels = out_channels

        self.res_blocks = nn.ModuleList(
            [ResBlock(in_channels) for _ in range(num_res)]
        )

        self.up_blocks = nn.ModuleList()
        for out_channels in up_channels:
            self.up_blocks.append(
                ConvBlock(
                    in_channels,
                    out_channels,
                    2,
                    up_sampling=True,
                    activation=nn.ReLU(),
                )
            )
            in_channels = out_channels

        self.final_block = ConvBlock(
            in_channels, final_channels, 1, activation=nn.Tanh()
        )

    def forward(self, x):
        x = self.initial_block(x)
        for block in self.down_blocks + self.res_blocks + self.up_blocks:
            x = block(x)
        return self.final_block(x)


def test():
    img_channels = 3
    img_size = 512
    x = torch.randn((7, img_channels, img_size, img_size))
    gen = StylizingNetwork()
    print(gen(x).shape)


if __name__ == "__main__":
    test()
