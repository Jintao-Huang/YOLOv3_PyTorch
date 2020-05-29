# Author: Jintao Huang
# Time: 2020-5-26

import torch
import torch.nn as nn
from .darknet53 import Conv2dBNLeakyReLU


class FeatureBlock(nn.Module):
    def __init__(self, in_channels, neck_channels, out_channels, norm_layer):
        assert len(neck_channels) == 2
        super(FeatureBlock, self).__init__()
        self.in_blocks = nn.Sequential(
            Conv2dBNLeakyReLU(in_channels, neck_channels[0], 1, 1, 0, False, norm_layer),
            Conv2dBNLeakyReLU(neck_channels[0], neck_channels[1], 3, 1, 1, False, norm_layer),
            Conv2dBNLeakyReLU(neck_channels[1], neck_channels[0], 1, 1, 0, False, norm_layer),
            Conv2dBNLeakyReLU(neck_channels[0], neck_channels[1], 3, 1, 1, False, norm_layer),
            Conv2dBNLeakyReLU(neck_channels[1], neck_channels[0], 1, 1, 0, False, norm_layer)
        )
        self.out_blocks = nn.Sequential(
            Conv2dBNLeakyReLU(neck_channels[0], neck_channels[1], 3, 1, 1, False, norm_layer),
            nn.Conv2d(neck_channels[1], out_channels, 1, 1, 0, bias=True)
        )

    def forward(self, x):
        x = self.in_blocks(x)
        x2 = self.out_blocks(x)
        return x, x2


class UpsampleBlock(nn.Sequential):
    def __init__(self, in_channels, norm_layer):
        assert in_channels % 2 == 0
        super(UpsampleBlock, self).__init__(
            Conv2dBNLeakyReLU(in_channels, in_channels // 2, 1, 1, 0, False, norm_layer),
            nn.UpsamplingNearest2d(scale_factor=2)
        )


class FPN(nn.Module):
    def __init__(self, in_channels_list, num_anchors, num_classes, norm_layer=None):
        """

        :param in_channels_list: P3, P4, P5
        :param out_channels: int. P3, P4, P5 的 out_channels
        :param norm_layer:
        """
        super(FPN, self).__init__()
        norm_layer = norm_layer or nn.BatchNorm2d
        self.num_anchors = num_anchors
        # create modules
        in_channels = 0
        in_channels_list = tuple(reversed(in_channels_list))  # P5, P4, P3
        neck_channels_list = ((512, 1024), (256, 512), (128, 256))  # P5, P4, P3
        feature_block_name = ["P5_block", "P4_block", "P3_block"]  # P5, P4, P3
        upsample_block_name = ["P5_to_P4", "P4_to_P3"]
        for i in range(3):
            in_channels = in_channels_list[i] + in_channels // 2
            setattr(self, feature_block_name[i],
                    FeatureBlock(in_channels, neck_channels_list[i], num_anchors * (5 + num_classes), norm_layer))
            if i != 2:
                in_channels = neck_channels_list[i][0]
                setattr(self, upsample_block_name[i], UpsampleBlock(in_channels, norm_layer))

    def forward(self, x):
        """

        :param x: P3, P4, P5
        :return: shape((N, F*H*W*A, 85)). F=3, A=3
        """
        x = list(x.values())
        feature_block_name = ["P5_block", "P4_block", "P3_block"]  # P5, P4, P3
        upsample_block_name = ["P5_to_P4", "P4_to_P3"]
        in_1 = list(reversed(x))  # P5, P4, P3
        in_2 = []  # P4_2, P3_2  # 第二个分支进入(The second branch goes in)
        output = []
        for i in range(3):
            if i == 0:
                x = in_1[i]
            else:
                upsample_block = getattr(self, upsample_block_name[i - 1])
                x = torch.cat((upsample_block(in_2[i - 1]), in_1[i]), dim=1)
            feature_block = getattr(self, feature_block_name[i])
            z, out = feature_block(x)
            output.append(out)
            in_2.append(z)
        del x, in_1, in_2, z, out  # 防止变量干扰(Prevent variable interference)
        output.reverse()  # P3_out, P4_out, P5_out.
        # shape(N, A*85, H, W) -> (N, -1(H*W*A), 85)
        for _ in range(len(output)):
            x = output.pop(0)
            x = x.permute((0, 2, 3, 1))  # N, H, W, C(A*85)
            x = torch.reshape(x, (x.shape[0], -1, x.shape[-1] // self.num_anchors))
            output.append(x)
        output = torch.cat(output, dim=1)
        output[:, :, :2] = torch.sigmoid(output[:, :, :2])  # x_reg, y_reg-> [0., 1.]
        output[:, :, 4:] = torch.sigmoid(output[:, :, 4:])  # reg_score, class_score -> [0., 1.]
        return output
