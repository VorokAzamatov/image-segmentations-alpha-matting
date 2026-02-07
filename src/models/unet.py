import torch
import torch.nn as nn

# -----------------------------
# Convolution block: two sequential Conv2d + ReLU
# -----------------------------
class ConvBlock(nn.Module):
    """
    Basic convolution block for UNet.
    Two 3x3 convolutions with ReLU.
    in_ch: input number of channels
    out_ch: output number of channels
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        out = self.relu(x)
        return out


# -----------------------------
# DownSample: ConvBlock + MaxPool
# -----------------------------
class DownSample(nn.Module):
    """
    Downsample UNet block.
    Returns:
        - x after MaxPool for the next level 
        - skip_connection for the skip path
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv_block = ConvBlock(in_ch, out_ch)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip_connection = self.conv_block(x)
        x = self.maxpool(skip_connection)
        return x, skip_connection


# -----------------------------
# UpSample: ConvTranspose + ConvBlock
# -----------------------------
class UpSample(nn.Module):
    """
    Upsample UNet block .
    Concatenation with skip connection and convolution to restore spatial dimensions.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.convT = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv_block = ConvBlock(in_ch, out_ch)

    def forward(self, x, skip_connection):
        x = self.convT(x)
        # конкатенация по channel dimension
        x = torch.cat([x, skip_connection], dim=1)
        out = self.conv_block(x)
        return out


# -----------------------------
# UNet
# -----------------------------
class UNet(nn.Module):
    """
    UNet model for segmentation.
    in_ch: number of input image channels (3 for RGB)
    num_cl: number of output channels (1 for mask)
    base_ch: base number of channels in the first level
    """
    def __init__(self, in_ch=3, num_cl=1, base_ch=64):
        super().__init__()
        # Downsampling path
        self.down1 = DownSample(in_ch, base_ch)
        self.down2 = DownSample(base_ch, base_ch*2)
        self.down3 = DownSample(base_ch*2, base_ch*4)
        self.down4 = DownSample(base_ch*4, base_ch*8)

        # Bottleneck
        self.bottleneck = ConvBlock(base_ch*8, base_ch*16)

        # Upsampling path
        self.up1 = UpSample(base_ch*16, base_ch*8)
        self.up2 = UpSample(base_ch*8, base_ch*4)
        self.up3 = UpSample(base_ch*4, base_ch*2)
        self.up4 = UpSample(base_ch*2, base_ch)

        # Output layer: 1x1 conv для получения 1 канала (маски)
        self.out_conv = nn.Conv2d(base_ch, num_cl, kernel_size=1)

    def forward(self, x):
        # Down path
        x, skip1 = self.down1(x)
        x, skip2 = self.down2(x)
        x, skip3 = self.down3(x)
        x, skip4 = self.down4(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Up path
        x = self.up1(x, skip4)
        x = self.up2(x, skip3)
        x = self.up3(x, skip2)
        x = self.up4(x, skip1)

        # Output
        out = self.out_conv(x)
        return out
