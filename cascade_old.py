# 3D-UNet model.

import torch
import torch.nn as nn
import math
from torch.nn import functional as F


class CReLU(nn.Module):
    def __init__(self):
        super(CReLU, self).__init__()

    def forward(self, x):
        return torch.cat((F.relu(x), F.relu(-x)), 1)


class up_conv3(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv3, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear'),
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


def conv_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        activation, )


def conv_trans_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        # nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear'),
        # nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=True),
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm3d(out_dim),
        activation, )


def max_pooling_3d():
    return nn.MaxPool3d(kernel_size=2, stride=2, padding=0, ceil_mode=True)


def conv_block_2_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        conv_block_3d(in_dim, out_dim, activation),
        nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim), )


class Attention_block(nn.Module):
    """
    Attention Block:
    - take g which is the spatially smaller signal (coarser scale), do a conv to get the same number of feature channels as x (bigger spatially)
    - do a conv on x to also get same feature channels (theta_x)
    - then, upsample g to be same size as x
    - add x and g (concat_xg)
    - relu, 1x1 conv, then sigmoid then upsample the final - this gives us attn coefficients
    """

    # F_l number of channels
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        # linear transformations

        self.W_g = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        input_size = x.size()

        g1 = self.W_g(g)
        x1 = self.W_x(x)
        W_x_size = x1.size()

        phi_g = F.interpolate(g1, size=W_x_size[2:], mode='trilinear', align_corners=False)
        psi = self.relu(phi_g + x1)
        psi = self.psi(psi)
        up_psi = F.interpolate(psi, size=input_size[2:], mode='trilinear', align_corners=False)
        out = x * up_psi

        return out


class cascade_UA_3D(nn.Module):
    def __init__(self, in_dim, out_dim, num_classes=3, num_filters=4):
        super(cascade_UA_3D, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filters = num_filters
        activation = nn.LeakyReLU(0.2, inplace=True)
        # 1) Stage
        # Down sampling
        self.down_1 = conv_block_2_3d(self.in_dim, self.num_filters, activation)
        self.pool_1 = max_pooling_3d()
        self.down_2 = conv_block_2_3d(self.num_filters, self.num_filters * 2, activation)
        self.pool_2 = max_pooling_3d()
        self.down_3 = conv_block_2_3d(self.num_filters * 2, self.num_filters * 4, activation)
        self.pool_3 = max_pooling_3d()
        self.down_4 = conv_block_2_3d(self.num_filters * 4, self.num_filters * 8, activation)
        self.pool_4 = max_pooling_3d()
        self.down_5 = conv_block_2_3d(self.num_filters * 8, self.num_filters * 16, activation)
        self.pool_5 = max_pooling_3d()

        # Bridge
        self.bridge1 = conv_block_2_3d(self.num_filters * 16, self.num_filters * 32, activation)

        # Up sampling
        self.trans_1 = conv_trans_block_3d(self.num_filters * 32, self.num_filters * 32, activation)
        self.up_1 = conv_block_2_3d(self.num_filters * 48, self.num_filters * 16, activation)
        self.trans_2 = conv_trans_block_3d(self.num_filters * 16, self.num_filters * 16, activation)
        self.up_2 = conv_block_2_3d(self.num_filters * 24, self.num_filters * 8, activation)
        self.trans_3 = conv_trans_block_3d(self.num_filters * 8, self.num_filters * 8, activation)
        self.up_3 = conv_block_2_3d(self.num_filters * 12, self.num_filters * 4, activation)
        self.trans_4 = conv_trans_block_3d(self.num_filters * 4, self.num_filters * 4, activation)
        self.up_4 = conv_block_2_3d(self.num_filters * 6, self.num_filters * 2, activation)
        self.trans_5 = conv_trans_block_3d(self.num_filters * 2, self.num_filters * 2, activation)
        self.up_5 = conv_block_2_3d(self.num_filters * 3, self.num_filters * 1, activation)

        # Output
        # self.out1 = conv_block_3d(self.num_filters, out_dim, activation)
        self.out1 = conv_block_3d(self.num_filters, out_dim, activation)

        # 2) Stage
        # Downsampling
        self.down_6 = conv_block_2_3d(self.in_dim, self.num_filters, activation)
        self.pool_6 = max_pooling_3d()
        self.down_7 = conv_block_2_3d(self.num_filters, self.num_filters * 2, activation)
        self.pool_7 = max_pooling_3d()
        self.down_8 = conv_block_2_3d(self.num_filters * 2, self.num_filters * 4, activation)
        self.pool_8 = max_pooling_3d()
        self.down_9 = conv_block_2_3d(self.num_filters * 4, self.num_filters * 8, activation)
        self.pool_9 = max_pooling_3d()
        self.down_10 = conv_block_2_3d(self.num_filters * 8, self.num_filters * 16, activation)
        self.pool_10 = max_pooling_3d()

        # Bridge
        self.bridge2 = conv_block_2_3d(self.num_filters * 16, self.num_filters * 32, activation)

        # Upsampling + Attention

        self.trans_6 = conv_trans_block_3d(self.num_filters * 32, self.num_filters * 32, activation)
        # self.Att6 = Attention_block(F_g=num_filters * 32, F_l=num_filters * 16, F_int=num_filters * 16)
        self.up_6 = conv_block_2_3d(self.num_filters * 48, self.num_filters * 16, activation)

        self.trans_7 = conv_trans_block_3d(self.num_filters * 16, self.num_filters * 16, activation)
        self.Att7 = Attention_block(F_g=num_filters * 16, F_l=num_filters * 8, F_int=num_filters * 8)
        self.up_7 = conv_block_2_3d(self.num_filters * 32, self.num_filters * 8, activation)

        self.trans_8 = conv_trans_block_3d(self.num_filters * 8, self.num_filters * 8, activation)
        self.Att8 = Attention_block(F_g=num_filters * 8, F_l=num_filters * 4, F_int=num_filters * 4)
        self.up_8 = conv_block_2_3d(self.num_filters * 16, self.num_filters * 4, activation)

        self.trans_9 = conv_trans_block_3d(self.num_filters * 4, self.num_filters * 4, activation)
        self.Att9 = Attention_block(F_g=num_filters * 4, F_l=num_filters * 2, F_int=num_filters * 2)
        self.up_9 = conv_block_2_3d(self.num_filters * 8, self.num_filters * 2, activation)

        self.trans_10 = conv_trans_block_3d(self.num_filters * 2, self.num_filters * 2, activation)
        self.Att10 = Attention_block(F_g=num_filters * 2, F_l=num_filters * 1, F_int=num_filters * 1)
        self.up_10 = conv_block_2_3d(self.num_filters * 4, self.num_filters * 1, activation)

        # output
        # self.out2 = conv_block_3d(self.num_filters, num_classes, activation)
        self.out2 = nn.Sequential(
            nn.Conv3d(self.num_filters, num_classes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_classes))

    def forward(self, x):
        # First stage
        # Down sampling
        down_1 = self.down_1(x)

        pool_1 = self.pool_1(down_1)
        # print(pool_1.shape)
        down_2 = self.down_2(pool_1)
        pool_2 = self.pool_2(down_2)

        down_3 = self.down_3(pool_2)
        pool_3 = self.pool_3(down_3)

        down_4 = self.down_4(pool_3)

        pool_4 = self.pool_4(down_4)

        down_5 = self.down_5(pool_4)

        pool_5 = self.pool_5(down_5)

        # Bridge
        bridge1 = self.bridge1(pool_5)
        # print(bridge1.shape)

        # Up sampling
        trans_1 = self.trans_1(bridge1)

        concat_1 = torch.cat([trans_1, down_5], dim=1)
        up_1 = self.up_1(concat_1)

        trans_2 = self.trans_2(up_1)

        concat_2 = torch.cat([trans_2, down_4], dim=1)
        up_2 = self.up_2(concat_2)

        trans_3 = self.trans_3(up_2)
        concat_3 = torch.cat([trans_3, down_3], dim=1)
        up_3 = self.up_3(concat_3)

        trans_4 = self.trans_4(up_3)
        concat_4 = torch.cat([trans_4, down_2], dim=1)
        up_4 = self.up_4(concat_4)

        trans_5 = self.trans_5(up_4)
        concat_5 = torch.cat([trans_5, down_1], dim=1)
        up_5 = self.up_5(concat_5)

        # Output
        out1 = self.out1(up_5)

        # Second stage

        # Downsampling
        down_6 = self.down_6(out1)
        pool_6 = self.pool_6(down_6)

        down_7 = self.down_7(pool_6)
        pool_7 = self.pool_7(down_7)

        down_8 = self.down_8(pool_7)  # -> [1, 16, 32, 32, 32]
        pool_8 = self.pool_8(down_8)  # -> [1, 16, 16, 16, 16]

        down_9 = self.down_9(pool_8)  # -> [1, 32, 16, 16, 16]
        pool_9 = self.pool_9(down_9)  # -> [1, 32, 8, 8, 8]

        down_10 = self.down_10(pool_9)  # -> [1, 64, 8, 8, 8]

        pool_10 = self.pool_10(down_10)  # -> [1, 64, 4, 4, 4]

        bridge2 = self.bridge2(pool_10)

        # Upsampling

        trans_6 = self.trans_6(bridge2)
        concat_6 = torch.cat([down_10, trans_6], dim=1)
        up_6 = self.up_6(concat_6)

        trans_7 = self.trans_7(up_6)
        Att7 = self.Att7(g=down_9, x=trans_7)
        concat_7 = torch.cat([Att7, trans_7], dim=1)
        up_7 = self.up_7(concat_7)

        trans_8 = self.trans_8(up_7)
        Att8 = self.Att8(g=down_8, x=trans_8)
        concat_8 = torch.cat([Att8, trans_8], dim=1)
        up_8 = self.up_8(concat_8)

        trans_9 = self.trans_9(up_8)
        Att9 = self.Att9(g=down_7, x=trans_9)
        concat_9 = torch.cat([Att9, trans_9], dim=1)
        up_9 = self.up_9(concat_9)

        trans_10 = self.trans_10(up_9)
        Att10 = self.Att10(g=down_6, x=trans_10)
        concat_10 = torch.cat([Att10, trans_10], dim=1)
        up_10 = self.up_10(concat_10)

        # Output
        out2 = self.out2(up_10)

        return out2

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p
