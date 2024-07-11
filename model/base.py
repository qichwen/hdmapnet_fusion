import torch
import torch.nn as nn

from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet18
import matplotlib.pyplot as plt
import torch.nn.functional as F



class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        #bevencoder : Up(64 + 256, 256, scale_factor=4)
        super().__init__()
        
        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            # in 320, out 256
            nn.BatchNorm2d(out_channels),
            # 256
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        # tf : x1 torch.Size([1, 256, 13, 25]) | x2 torch.Size([1, 64, 50, 100])
        # pred_nusc_camencode : torch.Size([6, 320, 4, 11]) | torch.Size([6, 112, 8, 22])
        # pred_nusc_bevencode x1: torch.Size([1, 256, 25, 50]) | x2 torch.Size([1, 64, 100, 200])
        

        x1 = self.up(x1)
        # tf : x1 torch.Size([1, 256, 52, 100]) | x2 torch.Size([1, 64, 50, 100])
        # pred_nusc_camencode : torch.Size([6, 320, 8, 22]) | torch.Size([6, 112, 8, 22])
        # pred_nusc_bevencode : x1 torch.Size([1, 256, 100, 200]) | x2 torch.Size([1, 64, 100, 200])
        if x1.size(2) != x2.size(2):
            x1 = F.interpolate(x1, size=(x2.size(2), x2.size(3)), mode='bilinear', align_corners=True)
        x1 = torch.cat([x2, x1], dim=1)
        # pred_nusc_camencode x1  torch.Size([6, 432, 8, 22])
        # pred_nusc_bevencode: torch.Size([1, 320, 100, 200])
        return self.conv(x1)


class CamEncode(nn.Module):
    def __init__(self, C):
        super(CamEncode, self).__init__()
        self.C = C

        self.trunk = EfficientNet.from_pretrained("efficientnet-b0")
        self.up1 = Up(320+112, self.C)

    def get_eff_depth(self, x):
        # adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.trunk._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints)+1)] = prev_x
            prev_x = x

        # Head
        endpoints['reduction_{}'.format(len(endpoints)+1)] = x
        x = self.up1(endpoints['reduction_5'], endpoints['reduction_4'])
        return x

    def forward(self, x):
        return self.get_eff_depth(x)


class BevEncode(nn.Module):
    def __init__(self, inC, outC, instance_seg=True, embedded_dim=16, direction_pred=True, direction_dim=37):
        super(BevEncode, self).__init__()
        trunk = resnet18(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.up1 = Up(64 + 256, 256, scale_factor=4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0),
        )

        self.instance_seg = instance_seg
        if instance_seg:
            self.up1_embedded = Up(64 + 256, 256, scale_factor=4)
            self.up2_embedded = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear',
                            align_corners=True),
                nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, embedded_dim, kernel_size=1, padding=0),
            )

        self.direction_pred = direction_pred
        if direction_pred:
            self.up1_direction = Up(64 + 256, 256, scale_factor=4)
            self.up2_direction = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear',
                            align_corners=True),
                nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, direction_dim, kernel_size=1, padding=0),
            )

    def forward(self, x):
        # torch.Size([1, 64, 100, 200])
        x = self.conv1(x)
        # torch.Size([1, 64, 50, 100])
        x = self.bn1(x)
        # torch.Size([1, 64, 50, 100])
        x = self.relu(x)
        # torch.Size([1, 64, 50, 100])
        x1 = self.layer1(x)
        # x1 torch.Size([1, 64, 50, 100])
        x = self.layer2(x1)
        # x torch.Size([1, 128, 25, 50])
        x2 = self.layer3(x)
        # x2 torch.Size([1, 256, 13, 25])
        
        #TODO: tf not matching issue! 
        x = self.up1(x2, x1)
        # x2: torch.Size([1, 256, 13, 25])
        # x1: torch.Size([1, 64, 50, 100])
        x = self.up2(x)

        if self.instance_seg:
            # x2 torch.Size([1, 256, 25, 50]) |  # x1: torch.Size([1, 64, 100, 200])
            x_embedded = self.up1_embedded(x2, x1)            
            x_embedded = self.up2_embedded(x_embedded)
        else:
            x_embedded = None

        if self.direction_pred:
            x_direction = self.up1_embedded(x2, x1)
            x_direction = self.up2_direction(x_direction)
        else:
            x_direction = None

        return x, x_embedded, x_direction
