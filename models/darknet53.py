# Author: Jintao Huang
# Time: 2020-5-26
import torch.nn as nn
import torchvision.transforms.transforms as trans
import torch
from .utils import load_state_dict_from_url

__all__ = ["preprocess", "Darknet53", "darknet53"]
model_url = "https://github.com/Jintao-Huang/Darknet53_PyTorch/releases/download/1.0/darknet53-26b80406.pth"


def preprocess(images, image_size):
    """预处理(preprocessing)

    :param images: List[PIL.Image]
    :param image_size: int
    :return: shape(N, C, H, W)
    """
    output = []
    trans_func = trans.Compose([
        trans.Resize(image_size),
        trans.ToTensor(),
        # trans.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    for image in images:
        if image.mode != "RGB":
            image = image.convert(mode="RGB")
        output.append(trans_func(image))
    return torch.stack(output, dim=0)


class Conv2dBNLeakyReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias, norm_layer):
        super(Conv2dBNLeakyReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            norm_layer(out_channels),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, norm_layer):
        super(ResidualBlock, self).__init__()
        assert in_channels % 2 == 0
        neck_channels = in_channels // 2
        self.conv1 = Conv2dBNLeakyReLU(in_channels, neck_channels, 1, 1, 0, False, norm_layer)
        self.conv2 = Conv2dBNLeakyReLU(neck_channels, in_channels, 3, 1, 1, False, norm_layer)

    def forward(self, x):
        inputs = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = inputs + x
        return x


class Darknet53(nn.Module):
    def __init__(self, num_classes=1000, norm_layer=None):
        super(Darknet53, self).__init__()
        norm_layer = norm_layer or nn.BatchNorm2d
        self.num_blocks = (1, 2, 8, 8, 4)

        self.conv = Conv2dBNLeakyReLU(3, 32, 3, 1, 1, False, norm_layer)
        in_channels = 32
        for i, num_repeat in enumerate(self.num_blocks):
            setattr(self, "layer%d" % (i + 1),
                    self._make_layers(in_channels, num_repeat, norm_layer))
            in_channels *= 2
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 用mean代替
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.conv(x)
        for i in range(len(self.num_blocks)):
            x = getattr(self, 'layer%d' % (i + 1))(x)
        x = torch.mean(x, dim=(2, 3))
        x = self.fc(x)
        return x

    @staticmethod
    def _make_layers(in_channels, num_repeat, norm_layer):
        layers = []
        out_channels = in_channels * 2
        layers.append(Conv2dBNLeakyReLU(in_channels, out_channels, 3, 2, 1, False, norm_layer))
        for i in range(num_repeat):
            layers.append(ResidualBlock(out_channels, norm_layer))
        return nn.Sequential(*layers)


def darknet53(pretrained=False, progress=True, num_classes=1000, norm_layer=None):
    model = Darknet53(num_classes, norm_layer)
    if pretrained:
        state_dict = load_state_dict_from_url(model_url, progress=progress)
        model.load_state_dict(state_dict)
    return model
