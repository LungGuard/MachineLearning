import torch as pt
import torch.nn as nn
import torch.nn.functional as F


class Conv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,pool_kernel_size=2,pool_stride=2, activation='relu'):
        super(Conv2DBlock, self).__init__()
        
        padding = kernel_size // 2
        
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)
        
        if activation == 'relu':
            self.act = nn.ReLU(inplace=True) # inplace חוסך בזיכרון GPU
        elif activation == 'leaky_relu':
            self.act = nn.LeakyReLU(inplace=True)
        else:
            self.act = nn.Identity()

    def forward(self, x):
            x = self.conv(x)
            x = self.bn(x)
            x = self.act(x)
            x = self.pool(x)
            return x