import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self, z_dim=100, img_channels=3, gfeat=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, gfeat*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(gfeat*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(gfeat*8, gfeat*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gfeat*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(gfeat*4, gfeat*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gfeat*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(gfeat*2, gfeat, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gfeat),
            nn.ReLU(True),

            nn.ConvTranspose2d(gfeat, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)
