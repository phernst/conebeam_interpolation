import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


# adapted from https://github.com/dchansen/ScatterNet/blob/master/ScatterNet.py


class SqueezeExcitation(nn.Module):
    def __init__(self, channels, squeeze_channels=None):
        if squeeze_channels is None:
            squeeze_channels = channels//8
        super().__init__()

        self.channels = channels
        self.fc1 = nn.Conv2d(channels, squeeze_channels, kernel_size=1)
        self.fc2 = nn.Conv2d(squeeze_channels, channels, kernel_size=1)

    def forward(self, x):
        out = F.avg_pool2d(x, x.size()[2:])
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        out = F.sigmoid(out)
        return x*out


class DownBlock(nn.Module):
    def __init__(self, inchannels, channels, activation=nn.ReLU,
                 batchnorm=False, squeeze=False, residual=True):
        super().__init__()
        self.residual = residual
        self.activation1 = activation()
        self.activation2 = activation()
        self.activation3 = activation()
        self.downconv = nn.Conv2d(
            inchannels, channels, kernel_size=2, stride=2, padding=1)

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        if batchnorm:
            self.bnorm1 = nn.BatchNorm2d(channels)
            self.bnorm2 = nn.BatchNorm2d(channels)
            self.bnorm3 = nn.BatchNorm2d(channels)
        if squeeze:
            self.squeeze = SqueezeExcitation(channels)
        else:
            self.squeeze = None
        self.batchnorm = batchnorm

    def forward(self, x):
        down = self.downconv(x)
        if self.batchnorm:
            down = self.bnorm1(down)
        down = self.activation1(down)
        out = self.conv1(down)
        if self.batchnorm:
            out = self.bnorm2(out)
        out = self.activation2(out)
        out = self.conv2(out)
        if self.batchnorm:
            out = self.bnorm3(out)
        if self.squeeze is not None:
            out = self.squeeze(out)
        if self.residual:
            out += down
        out = self.activation3(out)
        return out


class ResBlock(nn.Module):
    def __init__(self, inchannels, channels, activation=nn.ReLU,
                 batchnorm=False, squeeze=False, residual=True):
        super().__init__()
        self.residual = residual
        self.activation1 = activation()
        self.activation2 = activation()
        self.activation3 = activation()
        self.conv0 = nn.Conv2d(inchannels, channels, kernel_size=3, padding=1)
        if batchnorm:
            self.bnorm1 = nn.BatchNorm3d(channels)
            self.bnorm2 = nn.BatchNorm3d(channels)
            self.bnorm3 = nn.BatchNorm3d(channels)
        self.conv1 = nn.ConvTranspose2d(
            channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.ConvTranspose2d(
            channels, channels, kernel_size=3, padding=1)
        self.batchnorm = batchnorm
        if squeeze:
            self.squeeze = SqueezeExcitation(channels)
        else:
            self.squeeze = None

    def forward(self, x):
        up = self.conv0(x)
        if self.batchnorm:
            up = self.bnorm1(up)
        up = self.activation1(up)
        out = self.conv1(up)
        if self.batchnorm:
            out = self.bnorm2(out)
        out = self.activation2(out)
        out = self.conv2(out)
        if self.batchnorm:
            out = self.bnorm3(out)
        if self.squeeze is not None:
            out = self.squeeze(out)
        if self.residual:
            out += up

        out = self.activation3(out)
        return out


class UpBlock(nn.Module):
    def __init__(self, inchannels, channels, activation=nn.ReLU,
                 batchnorm=False, squeeze=False, residual=True):
        super().__init__()
        self.residual = residual
        self.activation1 = activation()
        self.activation2 = activation()
        self.activation3 = activation()
        self.upconv = nn.Conv2d(inchannels, channels, kernel_size=3, padding=1)
        if batchnorm:
            self.bnorm1 = nn.BatchNorm2d(channels)
            self.bnorm2 = nn.BatchNorm2d(channels)
            self.bnorm3 = nn.BatchNorm2d(channels)
        self.conv1 = nn.ConvTranspose2d(
            channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.ConvTranspose2d(
            channels, channels, kernel_size=3, padding=1)
        self.batchnorm = batchnorm
        if squeeze:
            self.squeeze = SqueezeExcitation(channels)
        else:
            self.squeeze = None

    def forward(self, x):
        up = F.interpolate(x, scale_factor=2)
        up = self.upconv(up)
        if self.batchnorm:
            up = self.bnorm1(up)
        up = self.activation1(up)
        out = self.conv1(up)
        if self.batchnorm:
            out = self.bnorm2(out)
        out = self.activation2(out)
        out = self.conv2(out)
        if self.batchnorm:
            out = self.bnorm3(out)
        if self.squeeze is not None:
            out = self.squeeze(out)
        if self.residual:
            out += up

        out = self.activation3(out)
        return out


class ConvertNet(nn.Module):
    def __init__(self, in_channels, activation=nn.ReLU):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1), activation(),
            nn.Conv2d(16, in_channels, kernel_size=3, padding=1), activation(),
        )

    def forward(self, x):
        return self.conv1(x)


class ScatterNet(nn.Module):
    def __init__(self, in_channels, layer_channels, batchnorm=False,
                 squeeze=False, skip_first=False, activation=nn.ReLU,
                 exp=False, residual=True):
        super().__init__()

        self.activation = activation
        self.conv1 = ConvertNet(in_channels, activation=activation)

        self.conv2 = ResBlock(
            in_channels, layer_channels[0], activation=activation,
            batchnorm=batchnorm, squeeze=squeeze, residual=residual)
        self.upblocks = nn.ModuleList()
        self.downblocks = nn.ModuleList()
        previous_channels = layer_channels[0]
        for channels in layer_channels[1:]:
            self.downblocks.append(DownBlock(
                previous_channels, channels, self.activation, batchnorm,
                squeeze, residual=residual))
            previous_channels = channels

        self.mix_block = nn.ModuleList()
        for channels in reversed(layer_channels[:-1]):
            self.mix_block.append(nn.Sequential(
                nn.Conv2d(channels*2, channels, kernel_size=1),
                self.activation()))
            self.upblocks.append(UpBlock(
                previous_channels, channels, self.activation, batchnorm,
                squeeze, residual=residual))
            previous_channels = channels

        self.d_conv_final = nn.ConvTranspose2d(
            layer_channels[0], 1, kernel_size=1, padding=0)
        self.skip_first = skip_first
        self.exp = exp

    def forward(self, x):
        if self.skip_first:
            level1 = x
        else:
            level1 = self.conv1(x)
        if self.exp:
            level1 = torch.exp(-level1)*2**(16)

        previous = self.conv2(level1)
        layers = [previous]
        for block in self.downblocks:
            previous = block(previous)
            layers.append(previous)

        layers = list(reversed(layers[:-1]))
        for block, shortcut, mixer in zip(self.upblocks, layers, self.mix_block):
            previous = block(previous)
            psize = previous.size()
            ssize = shortcut.size()
            if psize != ssize:
                diff = np.array(ssize, dtype=int) - np.array(psize, dtype=int)
                previous = F.pad(
                    previous, (0, int(diff[-1]), 0, int(diff[-2])),
                    mode="replicate")
            previous = torch.cat([previous, shortcut], dim=1)
            previous = mixer(previous)

        previous = self.d_conv_final(previous)
        if self.skip_first:
            return previous

        # if self.exp:
        #     previous = torch.clamp(level1-previous, min=1e-6)
        # else:
        #     previous = previous+level1
        return previous


def test_scatternet():
    from torchinfo import summary
    network = ScatterNet(2, [8, 16, 32, 64, 128, 256], activation=nn.LeakyReLU).cuda()
    summary(network, input_size=(1, 2, 256, 256))


if __name__ == '__main__':
    test_scatternet()
