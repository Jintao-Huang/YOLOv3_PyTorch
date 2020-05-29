# Author: Jintao Huang
# Time: 2020-5-28
import torch.nn as nn
import torch
from .backbone import Darknet53WithFPN
from .utils import PreProcess, PostProcess, load_state_dict_from_url
from .default_config import default_config
from .anchor import AnchorGenerator
from .loss import YOLOLoss

model_url = "https://github.com/Jintao-Huang/YOLOv3_PyTorch/releases/download/1.0/yolov3-6bd00218.pth"


class YOLOv3(nn.Module):
    def __init__(self, num_classes, config):
        """please use yolov3() """
        super().__init__()
        self.image_size = config['image_size']
        pretrained_backbone = config['pretrained_backbone']
        backbone_norm_layer = config['backbone_norm_layer']
        anchor_scales = config['anchor_scales']
        other_norm_layer = config['other_norm_layer']
        # ------------------------------
        self.preprocess = PreProcess()
        # anchor_scales 各层的len() 需要相同，这里不做检查
        num_anchors = len(anchor_scales[0])
        # yolo_head is combined in FPN
        self.backbone = Darknet53WithFPN(pretrained_backbone, num_anchors, num_classes,
                                         backbone_norm_layer, other_norm_layer)
        self.anchor_gen = AnchorGenerator(anchor_scales, (3, 4, 5))
        self.loss_fn = YOLOLoss()
        self.postprocess = PostProcess()

    def forward(self, image_list, targets=None, image_size=None, score_thresh=None, nms_thresh=None):
        """

        :param image_list: List[Tensor[C, H, W]]  [0., 1.]
        :param targets: Dict['labels': List[Tensor[NUMi]], 'boxes': List[Tensor[NUMi, 4]]]
            boxes: left, top, right, bottom
        :param image_size: int. 真实输入图片的大小
        :return: train模式: loss: Dict
                eval模式: result: Dict
        """

        assert isinstance(image_list, list) and isinstance(image_list[0], torch.Tensor)
        image_size = image_size or self.image_size
        # Notice: anchor_scales: (10, 13) - (373, 326).
        # Please adjust the resolution according to the specific situation
        image_size = min(1280, image_size // 32 * 32)  # 32整除
        image_list, targets = self.preprocess(image_list, targets, image_size)
        x = image_list.tensors
        features = self.backbone(x)
        anchors = self.anchor_gen(x)
        if targets is not None:
            if score_thresh is not None or nms_thresh is not None:
                print("Warning: no need to transfer score_thresh or nms_thresh")
            loss = self.loss_fn(features, anchors, targets)
            return loss
        else:
            score_thresh = score_thresh or 0.2
            nms_thresh = nms_thresh or 0.5
            result = self.postprocess(image_list, features, anchors, score_thresh, nms_thresh)
            return result


def yolov3(pretrained=False, num_classes=80, config=None):
    config = config or default_config
    if pretrained:
        config['pretrained_backbone'] = False
    # create modules
    model = YOLOv3(num_classes, config)
    if pretrained:
        state_dict = load_state_dict_from_url(model_url)
        model.load_state_dict(state_dict)

    return model
