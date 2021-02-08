import torch
import torch.nn as nn
from torch.nn import functional as F


class get_up_conv(nn.Module):
    """
    Up Convolution Block for Attention gate
    Args:
        in_ch(int): number of input channels
        out_ch (int): number of output channels
    """

    def __init__(self, in_ch, out_ch, activation):
        super(get_up_conv, self).__init__()

        self.activation = activation
        self.conv = convolution_2layers(in_ch + out_ch, out_ch, self.activation)
        self.up = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=True)
        # self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        x = torch.cat([inputs1, outputs2], 1)
        return self.conv(x)


def convolution_1layers(in_dim, out_dim, activation):
    """
        Basic convolutional module consisting of a Conv3d, non-linearity and optional batchnorm/groupnorm. """

    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        activation, )


def max_pooling_3d():
    return nn.MaxPool3d(kernel_size=2, stride=2, padding=0, ceil_mode=True)


def convolution_2layers(in_dim, out_dim, activation):
    """ A module consisting of two consecutive convolution layers (e.g. BatchNorm3d+ReLU+Conv3d)."""
    return nn.Sequential(
        convolution_1layers(in_dim, out_dim, activation),
        nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
    )


class Unet_3D(nn.Module):
    def __init__(self, in_dim, num_classes=3, feature_scale=8):
        super(Unet_3D, self).__init__()

        self.in_dim = in_dim
        self.feature_scale = feature_scale
        self.activation = nn.LeakyReLU(0.2, inplace=True)

        # It is common for a convolutional layer to learn from 32 to 512 filters in parallel for a given input. This
        # gives the model 32, or even 512, different ways of extracting features from an input, or many different
        # ways of both “learning to see” and after training, many different ways of “seeing” the input data.

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # 1) Stage
        # Down-sampling

        self.down_1 = convolution_2layers(self.in_dim, filters[0], self.activation)
        self.pool_1 = max_pooling_3d()
        self.down_2 = convolution_2layers(filters[0], filters[1], self.activation)
        self.pool_2 = max_pooling_3d()
        self.down_3 = convolution_2layers(filters[1], filters[2], self.activation)
        self.pool_3 = max_pooling_3d()
        self.down_4 = convolution_2layers(filters[2], filters[3], self.activation)
        self.pool_4 = max_pooling_3d()
        self.bridge = convolution_2layers(filters[3], filters[4], self.activation)

        # Up sampling
        self.up_4 = get_up_conv(filters[4], filters[3], self.activation)
        self.up_3 = get_up_conv(filters[3], filters[2], self.activation)
        self.up_2 = get_up_conv(filters[2], filters[1], self.activation)
        self.up_1 = get_up_conv(filters[1], filters[0], self.activation)

        # Output
        # self.out1 = convolution_1layers(self.num_filters, out_dim, self.activation)
        # self.out1 = convolution_1layers(filters[0], num_classes, self.activation)
        self.out1 = nn.Conv3d(filters[0], num_classes, 1)

    def forward(self, x):
        # 1 Stage

        # Down-sampling
        down_1 = self.down_1(x)
        pool_1 = self.pool_1(down_1)

        down_2 = self.down_2(pool_1)
        pool_2 = self.pool_2(down_2)

        down_3 = self.down_3(pool_2)
        pool_3 = self.pool_3(down_3)

        down_4 = self.down_4(pool_3)
        pool_4 = self.pool_4(down_4)

        # center
        bridge1 = self.bridge(pool_4)

        # Up sampling

        up_4 = self.up_4(down_4, bridge1)
        up_3 = self.up_3(down_3, up_4)
        up_2 = self.up_2(down_2, up_3)
        up_1 = self.up_1(down_1, up_2)

        # Output
        out1 = self.out1(up_1)

        return out1
