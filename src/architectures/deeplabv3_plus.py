import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class Atrous_Convolution(torch.nn.Module):
    def __init__(self, in_channels, kernel_size, dilation_rate, padding, out_channels=256):
        super().__init__()

        self.conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation_rate,
            bias=False
        )
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ASPP(torch.nn.Module):
    def __init__(self, in_channels=1024):
        super().__init__()

        self.conv_1x1 = Atrous_Convolution(in_channels, 1, 1, 0)
        self.conv_6x6 = Atrous_Convolution(in_channels, 3, 6, 6)
        self.conv_12x12 = Atrous_Convolution(in_channels, 3, 12, 12)
        self.conv_18x18 = Atrous_Convolution(in_channels, 3, 18, 18)

        self.image_pool = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Conv2d(in_channels, 256, kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True)
        )

        self.project = torch.nn.Sequential(
            torch.nn.Conv2d(256 * 5, 256, kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5)
        )

    def forward(self, x):
        h, w = x.shape[2:]

        b1 = self.conv_1x1(x)
        b2 = self.conv_6x6(x)
        b3 = self.conv_12x12(x)
        b4 = self.conv_18x18(x)

        b5 = self.image_pool(x)
        b5 = torch.nn.functional.interpolate(b5, size=(h, w), mode='bilinear', align_corners=False)

        x = torch.cat([b1, b2, b3, b4, b5], dim=1)
        return self.project(x)


class ResNetBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()

        resnet = models.resnet101(weights="DEFAULT")

        self.layer0 = torch.nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

    def forward(self, x):
        x = self.layer0(x)
        low_level = self.layer1(x)
        x = self.layer2(low_level)
        high_level = self.layer3(x)

        return high_level, low_level
    
class Decoder(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.low_level_conv = torch.nn.Sequential(
            torch.nn.Conv2d(256, 48, kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(48),
            torch.nn.ReLU(inplace=True)
        )

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, x, low_level):
        low_level = self.low_level_conv(low_level)

        x = torch.nn.functional.interpolate(
            x, size=low_level.shape[2:], mode='bilinear', align_corners=False
        )

        x = torch.cat([x, low_level], dim=1)
        return self.conv(x)
    

class DeepLabV3Plus(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.backbone = ResNetBackbone()
        self.aspp = ASPP(in_channels=1024)  
        self.decoder = Decoder(num_classes)

    def forward(self, x):
        input_size = x.shape[2:]

        high_level, low_level = self.backbone(x)

        x = self.aspp(high_level)
        x = self.decoder(x, low_level)

        x = torch.nn.functional.interpolate(
            x, size=input_size, mode='bilinear', align_corners=False
        )

        return x