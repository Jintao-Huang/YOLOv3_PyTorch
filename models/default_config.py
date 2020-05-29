# Author: Jintao Huang
# Time: 2020-5-27

import torch.nn as nn
from .utils import FrozenBatchNorm2d
import numpy as np

default_config = {
    "image_size": 416,
    # backbone
    "pretrained_backbone": True,
    "backbone_norm_layer": FrozenBatchNorm2d,
    # anchor:
    "anchor_scales": [  # yolov3  (H, W)
        np.array([[13, 10], [30, 16], [23, 33]]),
        np.array([[61, 30], [45, 62], [119, 59]]),
        np.array([[90, 116], [198, 156], [326, 373]]),
    ],
    # other:
    "other_norm_layer": nn.BatchNorm2d,
}
