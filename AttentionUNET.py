import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MaxPool2d


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class AttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)

        self.theta = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.phi = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.g = nn.Conv2d(out_channels, out_channels, kernel_size=1)

        self.attn_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.attn_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x

        x = self.conv(x)
        x = self.bn(x)

        theta = self.theta(x)
        phi = self.phi(x)
        g = self.g(x)

        theta = theta.view(theta.shape[0], theta.shape[1], -1)
        phi = phi.view(phi.shape[0], phi.shape[1], -1)
        g = g.view(g.shape[0], g.shape[1], -1)

        attn = F.softmax(torch.bmm(theta.permute(0, 2, 1), phi), dim=2)
        attn = torch.bmm(g, attn.permute(0, 2, 1))
        attn = attn.view(x.shape)

        attn = self.attn_conv(attn)
        attn = self.attn_bn(attn)

        x = x + attn
        x = F.relu(x)

        return x + residual


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x, skip_x):
        x = self.up(x)
        x = torch.cat([x, skip_x], axis=1)
        x = self.conv(x)
        return x


class AttU_Net(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(AttU_Net, self).__init__()

        features = init_features
        self.encoder1 = ConvBlock(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = ConvBlock(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = ConvBlock(features * 2, features * 4)
        self.pool3 = nn

        MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = ConvBlock(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.center = ConvBlock(features * 8, features * 16)

        self.decoder4 = UpBlock(features * 16, features * 8)
        self.attention4 = AttentionBlock(features * 16, features * 8)
        self.decoder3 = UpBlock(features * 8, features * 4)
        self.attention3 = AttentionBlock(features * 8, features * 4)
        self.decoder2 = UpBlock(features * 4, features * 2)
        self.attention2 = AttentionBlock(features * 4, features * 2)
        self.decoder1 = UpBlock(features * 2, features)
        self.attention1 = AttentionBlock(features * 2, features)

        self.output = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x):
        skip1 = self.encoder1(x)
        x = self.pool1(skip1)

        skip2 = self.encoder2(x)
        x = self.pool2(skip2)

        skip3 = self.encoder3(x)
        x = self.pool3(skip3)

        skip4 = self.encoder4(x)
        x = self.pool4(skip4)

        x = self.center(x)

        x = self.decoder4(x, skip4)
        x = self.attention4(x)

        x = self.decoder3(x, skip3)
        x = self.attention3(x)

        x = self.decoder2(x, skip2)
        x = self.attention2(x)

        x = self.decoder1(x, skip1)
        x = self.attention1(x)

        x = self.output(x)

        return x
