import torch
import torch.nn.functional as F


class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, apply_pool=True):
        super().__init__()

        self.conv_block = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=input_size,
                out_channels=output_size,
                kernel_size=kernel_size,
                stride=stride,
                padding=1),
            torch.nn.BatchNorm2d(output_size),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(
                in_channels=output_size,
                out_channels=output_size,
                kernel_size=kernel_size,
                stride=stride,
                padding=1),
            torch.nn.BatchNorm2d(output_size),
            torch.nn.ReLU(inplace=True)

        )

        self.max_pool = torch.nn.MaxPool2d(kernel_size=2, stride=2) if apply_pool else None

    def forward(self, x):

        x = self.conv_block(x)
        skip_connection = x
        if self.max_pool is not None:
            x = self.max_pool(x)

        return x, skip_connection

class DeconvBlock(torch.nn.Module):
    def __init__(self, input_channels, skip_channels, output_channels, kernel_size=3, stride=1):
        super().__init__()

        self.deconv_block = torch.nn.ConvTranspose2d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=2,
            stride=2)

        self.conv_block = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=output_channels + skip_channels,
                out_channels=output_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=1),
            torch.nn.BatchNorm2d(output_channels),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(
                in_channels=output_channels,
                out_channels=output_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=1),
            torch.nn.BatchNorm2d(output_channels),
            torch.nn.ReLU(inplace=True)
        )
    def forward(self, x, external_input):
        x = self.deconv_block(x)
        x = torch.cat((x, external_input), dim=1)
        x = self.conv_block(x)

        return x

class UNet(torch.nn.Module):
    def __init__(self, input_size=3, num_classes=13):
        super().__init__()

        # Input block
        self.input_block = ConvBlock(input_size=input_size, output_size=64)

        # Downsample block
        self.downsample_block1 = ConvBlock(input_size=64, output_size=128)
        self.downsample_block2 = ConvBlock(input_size=128, output_size=256)
        self.downsample_block3 = ConvBlock(input_size=256, output_size=512)
        self.downsample_block4 = ConvBlock(input_size=512, output_size=1024, apply_pool=False)

        # Upsample block
        self.upsample_block1 = DeconvBlock(input_channels=1024, skip_channels=512, output_channels=512)
        self.upsample_block2 = DeconvBlock(input_channels=512, skip_channels=256, output_channels=256)
        self.upsample_block3 = DeconvBlock(input_channels=256, skip_channels=128, output_channels=128)
        self.upsample_block4 = DeconvBlock(input_channels=128, skip_channels=64, output_channels=64)

        self.conv_1x1 = torch.nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)

    def forward(self, x):

        x, skip_connection0 = self.input_block(x)

        # Downsample step
        x, skip_connection1 = self.downsample_block1(x)
        x, skip_connection2 = self.downsample_block2(x)
        x, skip_connection3 = self.downsample_block3(x)
        x, skip_connection4 = self.downsample_block4(x)

        # Upsample step
        x = self.upsample_block1(x, skip_connection3)
        x = self.upsample_block2(x, skip_connection2)
        x = self.upsample_block3(x, skip_connection1)
        x = self.upsample_block4(x, skip_connection0)

        out = self.conv_1x1(x)

        return out