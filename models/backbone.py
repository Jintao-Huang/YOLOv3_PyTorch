# Author: Jintao Huang
# Time: 2020-5-26
import torch
from .utils import IntermediateLayerGetter
from .darknet53 import darknet53
import torch.nn as nn
from collections import OrderedDict
from .fpn import FPN


class Darknet53WithFPN(nn.Sequential):
    def __init__(self, pretrained_backbone, out_anchors, num_classes, backbone_norm_layer, fpn_norm_layer):
        backbone = darknet53(pretrained_backbone, norm_layer=backbone_norm_layer)
        return_layers = {"layer3": "P3", "layer4": "P4", "layer5": "P5"}
        in_channels_list = (256, 512, 1024)  # P3 P4 P5
        super(Darknet53WithFPN, self).__init__(OrderedDict({
            "body": IntermediateLayerGetter(backbone, return_layers),
            "fpn": FPN(in_channels_list, out_anchors, num_classes, fpn_norm_layer)
        }))
