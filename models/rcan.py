import math

import torch
from torch import nn, Tensor

class RCAN(nn.Module):
    def __init__(
            self,
            scale: int,
            rgb_mean: tuple = None,
    ) -> None:
        super(RCAN, self).__init__()
        if rgb_mean is None:
            rgb_mean = [0.4488, 0.4371, 0.4040]

        # The first layer of convolutional layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

        # Feature extraction backbone
        trunk = []
        for _ in range(10):
            trunk.append(ResidualGroup(64, 16, 20))
        self.trunk = nn.Sequential(*trunk)

        # After the feature extraction network, reconnect a layer of convolutional blocks
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # Upsampling convolutional layer.
        upsampling = []
        if scale == 2 or scale == 4 or scale == 8:
            for _ in range(int(math.log(scale, 2))):
                upsampling.append(UpsampleBlock(64, 2))
        elif scale == 3:
            upsampling.append(UpsampleBlock(64, 3))
        self.upsampling = nn.Sequential(*upsampling)

        # Output layer.
        self.conv3 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

        self.register_buffer("mean", Tensor(rgb_mean).view(1, 3, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        x = x.sub_(self.mean).mul_(1.)

        conv1 = self.conv1(x)
        x = self.trunk(conv1)
        x = self.conv2(x)
        x = torch.add(x, conv1)
        x = self.upsampling(x)
        x = self.conv3(x)

        x = x.div_(1.).add_(self.mean)

        return x


class ChannelAttentionLayer(nn.Module):
    def __init__(self, channel: int, reduction: int):
        super(ChannelAttentionLayer, self).__init__()
        self.channel_attention_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, (1, 1), (1, 1), (0, 0)),
            nn.ReLU(True),
            nn.Conv2d(channel // reduction, channel, (1, 1), (1, 1), (0, 0)),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.channel_attention_layer(x)

        out = torch.mul(out, x)

        return out


class ResidualChannelAttentionBlock(nn.Module):
    def __init__(self, channel: int, reduction: int):
        super(ResidualChannelAttentionBlock, self).__init__()
        self.residual_channel_attention_block = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            ChannelAttentionLayer(channel, reduction),
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.residual_channel_attention_block(x)

        out = torch.add(out, identity)

        return out


class ResidualGroup(nn.Module):
    def __init__(self, channel: int, reduction: int, num_rcab: int):
        super(ResidualGroup, self).__init__()
        residual_group = []

        for _ in range(num_rcab):
            residual_group.append(ResidualChannelAttentionBlock(channel, reduction))
        residual_group.append(nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1))

        self.residual_group = nn.Sequential(*residual_group)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.residual_group(x)

        out = torch.add(out, identity)

        return out


class UpsampleBlock(nn.Module):
    def __init__(self, channels: int, upscale_factor: int) -> None:
        super(UpsampleBlock, self).__init__()
        self.upsample_block = nn.Sequential(
            nn.Conv2d(channels, channels * upscale_factor * upscale_factor, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.upsample_block(x)

        return x


