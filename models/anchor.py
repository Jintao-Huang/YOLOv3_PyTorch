# Author: Jintao Huang
# Time: 2020-5-28
import torch.nn as nn
import torch


class AnchorGenerator(nn.Module):
    def __init__(self, scales, pyramid_levels=None):
        """

        :param scales: tuple[tuple(H, W)].
        :param pyramid_levels: tuple[int]
        """
        super(AnchorGenerator, self).__init__()
        pyramid_levels = pyramid_levels or (3, 4, 5)
        self.scales = scales
        self.strides = [2 ** i for i in pyramid_levels]
        self.image_size = None  # int
        self.anchors = None

    def forward(self, x):
        """

        :param x: (images)Tensor[N, 3, H, W]. need: {.shape, .device, .dtype}
        :return: anchors[X(F*H*W*A), 5]. (cell_x, cell_y, stride, width, height)
        """
        image_size, dtype, device = x.shape[3], x.dtype, x.device
        if self.image_size == image_size:  # Anchors has been generated
            return self.anchors.to(device, dtype, copy=False)  # default: False
        else:
            self.image_size = image_size

        anchors_all = []
        for stride, scales in zip(self.strides, self.scales):
            anchors_level = []
            for scale in scales:
                if image_size % stride != 0:
                    raise ValueError('input size must be divided by the stride.')
                anchor_h, anchor_w = scale
                cell_x = torch.arange(0, image_size // stride, dtype=dtype, device=device)
                cell_y = torch.arange(0, image_size // stride, dtype=dtype, device=device)
                cell_y, cell_x = torch.meshgrid(cell_y, cell_x)
                cell_x = cell_x.reshape(-1)
                cell_y = cell_y.reshape(-1)
                strides = torch.full_like(cell_x, stride, dtype=dtype, device=device)
                anchor_w = torch.full_like(cell_x, anchor_w, dtype=dtype, device=device)
                anchor_h = torch.full_like(cell_x, anchor_h, dtype=dtype, device=device)

                # shape(X, 5)
                anchors = torch.stack([cell_x, cell_y, strides, anchor_w, anchor_h], dim=-1)
                anchors_level.append(anchors)

            anchors_level = torch.stack(anchors_level, dim=1).reshape(-1, 5)  # shape(X, A, 5) -> (-1, 5)
            anchors_all.append(anchors_level)
        self.anchors = torch.cat(anchors_all, dim=0)  # shape(-1, 5)
        return self.anchors
