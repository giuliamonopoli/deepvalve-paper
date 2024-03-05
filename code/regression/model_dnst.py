import torch
import torch.nn as nn

# def conv_block(inputs, filters):
#     return nn.Sequential(
#         nn.Conv2d(inputs, filters, kernel_size=3, padding=1),
#         nn.BatchNorm2d(filters),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(filters, filters, kernel_size=3, padding=1),
#         nn.BatchNorm2d(filters),
#         nn.ReLU(inplace=True)
#     )

# class UNetDSNT(nn.Module):
#     def __init__(self, input_channels = 1):
#         super(UNetDSNT, self).__init__()

#        # Contracting path
#         self.conv_down1 = conv_block(input_channels, 8)
#         self.pool_down1 = nn.MaxPool2d(kernel_size=4, stride=4)
#         self.conv_down2 = conv_block(8, 16)
#         self.pool_down2 = nn.MaxPool2d(kernel_size=2)
#         self.conv_down3 = conv_block(16, 32)
#         self.pool_down3 = nn.MaxPool2d(kernel_size=2)
#         self.conv_down4 = conv_block(32, 64)
#         self.pool_down4 = nn.MaxPool2d(kernel_size=2)

#         # Bottleneck
#         self.bottleneck = conv_block(64, 64)
#         self.upsample_bottleneck = nn.Upsample(scale_factor=2)

#         # Expanding path
#         self.conv_up4 = conv_block(128, 64)
#         self.upsample4 = nn.Upsample(scale_factor=2)
#         self.conv_up3 = conv_block(64, 32)
#         self.upsample3 = nn.Upsample(scale_factor=2)
#         self.conv_up2 = conv_block(32, 16)

#         # Heatmap and coordinates prediction
#         self.heatmap = nn.Conv2d(16, 20, kernel_size=3, padding=1)
#         self.softmax = nn.Softmax(dim=1)
#         self.coordinates = self.spatial_to_coordinates
#     @staticmethod
#     def spatial_to_coordinates(inputs):
#         px_range = torch.arange(1, 65, dtype=torch.float32) / 64.0
#         x, y = torch.meshgrid(px_range, px_range)
#         x = torch.sum(inputs * x[None, ...], dim=(1, 2))
#         y = torch.sum(inputs * y[None, ...], dim=(1, 2))
#         return torch.stack((y, x), dim=-1)

#     def forward(self, x):
#         # Contracting path

#         conv_down1 = self.conv_down1(x)
#         print("c1",conv_down1.shape)
#         pool_down1 = self.pool_down1(conv_down1)
#         print("d1",pool_down1.shape)
#         conv_down2 = self.conv_down2(pool_down1)
#         print("c2",conv_down2.shape)
#         pool_down2 = self.pool_down2(conv_down2)
#         print("d2",pool_down2.shape)
#         conv_down3 = self.conv_down3(pool_down2)
#         print("c3",conv_down3.shape)
#         pool_down3 = self.pool_down3(conv_down3)
#         print("d3",pool_down3.shape)
#         conv_down4 = self.conv_down4(pool_down3)
#         print("c4",conv_down4.shape)
#         pool_down4 = self.pool_down4(conv_down4)
#         print("d4",pool_down4.shape)

#         # Bottleneck
#         bottleneck = self.bottleneck(pool_down4)
#         upsample_bottleneck = self.upsample_bottleneck(bottleneck)
#         print(upsample_bottleneck.shape,conv_down4.shape )

#         # Expanding path
#         concat4 = torch.cat(( upsample_bottleneck, conv_down4), dim=1)
#         conv_up4 = self.conv_up4(concat4)
#         upsample4 = self.upsample4(conv_up4)
#         concat3 = torch.cat((conv_down3, upsample4), dim=1)
#         conv_up3 = self.conv_up3(concat3)
#         upsample3 = self.upsample3(conv_up3)
#         concat2 = torch.cat((conv_down2, upsample3), dim=1)
#         conv_up2 = self.conv_up2(concat2)

#         # Heatmap
#         heatmap = self.heatmap(conv_up2)
#         heatmap = self.softmax(heatmap)

#         # Coordinates prediction
#         coordinates = self.coordinates(heatmap)

#         return coordinates


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.down_sample = nn.MaxPool2d(2)

    def forward(self, x):
        skip_out = self.double_conv(x)
        down_out = self.down_sample(skip_out)
        return (down_out, skip_out)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up_sample_mode):
        super(UpBlock, self).__init__()
        if up_sample_mode == "conv_transpose":
            self.up_sample = nn.ConvTranspose2d(
                in_channels - out_channels,
                in_channels - out_channels,
                kernel_size=2,
                stride=2,
            )
        elif up_sample_mode == "bilinear":
            self.up_sample = nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            )
        else:
            raise ValueError(
                "Unsupported `up_sample_mode` (can take one of `conv_transpose` or `bilinear`)"
            )
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, down_input, skip_input):
        x = self.up_sample(down_input)
        x = torch.cat([x, skip_input], dim=1)
        return self.double_conv(x)


class UNetDSNT(nn.Module):
    def __init__(self, out_classes=40, up_sample_mode="conv_transpose"):
        super(UNetDSNT, self).__init__()
        self.up_sample_mode = up_sample_mode
        # Downsampling Path
        self.down_conv1 = DownBlock(1, 64)
        self.down_conv2 = DownBlock(64, 128)
        self.down_conv3 = DownBlock(128, 256)
        self.down_conv4 = DownBlock(256, 512)
        # Bottleneck
        self.double_conv = DoubleConv(512, 1024)
        # Upsampling Path
        self.up_conv4 = UpBlock(512 + 1024, 512, self.up_sample_mode)
        self.up_conv3 = UpBlock(256 + 512, 256, self.up_sample_mode)
        self.up_conv2 = UpBlock(128 + 256, 128, self.up_sample_mode)
        self.up_conv1 = UpBlock(128 + 64, 64, self.up_sample_mode)
        # Final Convolution
        self.conv_last = nn.Conv2d(64, out_classes, kernel_size=1)

    def forward(self, x):
        x, skip1_out = self.down_conv1(x)
        x, skip2_out = self.down_conv2(x)
        x, skip3_out = self.down_conv3(x)
        x, skip4_out = self.down_conv4(x)
        x = self.double_conv(x)
        x = self.up_conv4(x, skip4_out)
        x = self.up_conv3(x, skip3_out)
        x = self.up_conv2(x, skip2_out)
        x = self.up_conv1(x, skip1_out)
        x = self.conv_last(x)
        return x
