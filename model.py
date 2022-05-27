import torch
import torch.nn as nn
from blocks import *
from functools import reduce
import numpy as np
import cv2
import time
import numpy as np
import os

class RDB(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB, self).__init__()
        Cin = inChannels
        G = growRate

        self.conv1 = nn.Conv2d(Cin, G, kSize, padding=(kSize -1 )//2, stride=1)
        self.conv2 = nn.Conv2d(Cin + G, G, kSize, padding=(kSize -1 )//2, stride=1)
        self.conv3 = nn.Conv2d(Cin + 2 * G, G, kSize, padding=(kSize -1 )//2, stride=1)
        self.conv4 = nn.Conv2d(Cin + 3 * G, G, kSize, padding=(kSize - 1) // 2, stride=1)
        self.conv5 = nn.Conv2d(Cin + 4 * G, G, kSize, padding=(kSize - 1) // 2, stride=1)
        self.conv6 = nn.Conv2d(Cin + 5 * G, Cin, kSize, padding=(kSize - 1) // 2, stride=1)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.act(self.conv1(x))
        x2 = self.act(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.act(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.act(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.act(self.conv5(torch.cat((x, x1, x2, x3, x4), 1)))
        x6 = self.conv6(torch.cat((x, x1, x2, x3, x4, x5), 1))

        return 0.1 * x6 + x

class WRDB(nn.Module):
    def __init__(self, num_features, growRate, kSize=3):
        super(WRDB, self).__init__()
        self.RDB1 = RDB(num_features, growRate, kSize)
        self.RDB2 = RDB(num_features, growRate, kSize)
        self.RDB3 = RDB(num_features, growRate, kSize)

        self.conv0= nn.Conv2d(num_features, num_features, kSize, padding=(kSize -1 )//2, stride=1)
        self.conv1_1 = nn.Conv2d(in_channels=num_features, out_channels=num_features,
                               kernel_size=1, padding=0, stride=1, groups=num_features)
        self.conv1_2 = nn.Conv2d(in_channels=num_features, out_channels=num_features,
                                 kernel_size=1, padding=0, stride=1, groups=num_features)
        self.conv1_3 = nn.Conv2d(in_channels=num_features, out_channels=num_features,
                                 kernel_size=1, padding=0, stride=1, groups=num_features)

        self.conv2_1 = nn.Conv2d(in_channels=num_features, out_channels=num_features,
                               kernel_size=1, padding=0, stride=1, groups=num_features)
        self.conv2_2 = nn.Conv2d(in_channels=num_features, out_channels=num_features,
                                 kernel_size=1, padding=0, stride=1, groups=num_features)

        self.conv3_1 = nn.Conv2d(in_channels=num_features, out_channels=num_features,
                               kernel_size=1, padding=0, stride=1, groups=num_features)

        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
    def forward(self, x):
        x1_0 = self.act(self.conv0(x))
        x1_1 = self.conv1_1(x1_0)
        x2_0 = self.RDB1(x1_0)
        x2 = x2_0 + x1_1
        x2_1 = self.conv2_1(x2)
        x1_2 = self.conv1_2(x1_0)
        x3_0 = self.RDB2(x2)
        x3 = x2_1 + x1_2 + x3_0
        x1_3 = self.conv1_3(x1_0)
        x2_2 = self.conv2_2(x2)
        x3_1 = self.conv3_1(x3)
        x4 = self.RDB3(x3)
        out = x1_3 + x2_2 + x3_1 + x4
        out = 0.1 * out + x
        return out

class Upsampler(nn.Module):
    def __init__(self, num_features, out_channels, kSize, scale):
        super(Upsampler, self).__init__()
        # Up-sampling net
        if scale == 2 or scale == 3:
            self.UPNet = nn.Sequential(*[
                nn.Conv2d(num_features, num_features * scale * scale, kSize, padding=(kSize - 1) // 2, stride=1),
                nn.PixelShuffle(scale),
                nn.Conv2d(num_features, out_channels, kSize, padding=(kSize - 1) // 2, stride=1)
            ])
        elif scale == 4:
            self.UPNet = nn.Sequential(*[
                nn.Conv2d(num_features, num_features * 4, kSize, padding=(kSize - 1) // 2, stride=1),
                nn.PixelShuffle(2),
                nn.Conv2d(num_features, num_features * 4, kSize, padding=(kSize - 1) // 2, stride=1),
                nn.PixelShuffle(2),
                nn.Conv2d(num_features, out_channels, kSize, padding=(kSize - 1) // 2, stride=1)
            ])
        elif scale == 8:
            self.UPNet = nn.Sequential(*[
            nn.Conv2d(num_features, num_features * 4, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.PixelShuffle(2),
            nn.Conv2d(num_features, num_features * 4, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.PixelShuffle(2),
            nn.Conv2d(num_features, num_features * 4, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.PixelShuffle(2),
            nn.Conv2d(num_features, out_channels, kSize, padding=(kSize - 1) // 2, stride=1)
        ])
        else:
            raise ValueError("scale must be 2 or 3 or 4.")

    def forward(self, x):
        out = self.UPNet(x)

        return out

class AsyCA(nn.Module):
    def __init__(self, num_features, ratio):
        super(AsyCA, self).__init__()
        self.out_channels = num_features
        self.conv_init = nn.Conv2d(num_features * 2, num_features, kernel_size=1, padding=0, stride=1)
        self.conv_dc = nn.Conv2d(num_features, num_features // ratio, kernel_size=1, padding=0, stride=1)
        self.conv_ic = nn.Conv2d(num_features // ratio, num_features * 2, kernel_size=1, padding=0, stride=1)
        self.act = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2):
        batch_size = x1.size(0)

        feat_init = torch.cat((x1, x2), 1)
        feat_init = self.conv_init(feat_init)
        fea_avg = self.avg_pool(feat_init)
        feat_ca = self.conv_dc(fea_avg)
        feat_ca = self.conv_ic(self.act(feat_ca))

        a_b = feat_ca.reshape(batch_size, 2, self.out_channels, -1)

        a_b = self.softmax(a_b)
        # print(a_b[0,0,0,0],)
        a_b = list(a_b.chunk(2, dim=1))  # split to a and b
        a_b = list(map(lambda x1: x1.reshape(batch_size, self.out_channels, 1, 1), a_b))
        self.V1 =V1= a_b[0] * x1
        self.V2 =V2= a_b[1] * x2
        V = V1 + V2
        return V
        
class WaterFusion(nn.Module):
    def __init__(self, in_channels, out_channels, num_features, growthrate):
        super(WaterFusion, self).__init__()
        kSize = 3
        ratio = 4

        self.feat_conv1 = nn.Conv2d(in_channels, num_features, kSize, padding=(kSize - 1) // 2, stride=1)
        self.feat_conv2 = nn.Conv2d(in_channels, num_features, kSize, padding=(kSize - 1) // 2, stride=1)
        self.RDB0 = RDB(num_features, growthrate, kSize)
        self.RDB1 = RDB(num_features, growthrate, kSize)
        self.RDB2 = RDB(num_features, growthrate, kSize)
        # self.RDB3 = RDB(num_features, growthrate, kSize)

        self.AsyCA1 = AsyCA(num_features, ratio)
        # self.AsyCA2 = AsyCA(num_features, ratio)
        # self.AsyCA3 = AsyCA(num_features, ratio)

        self.out_conv = nn.Conv2d(num_features, out_channels, 1, padding=0, stride=1)

    	# self.tanh = nn.Tanh()

    def forward(self, pre1, pre2, w=None):
        x1 = self.feat_conv1(pre1)
        self.x1 = x1 = self.RDB0(x1)

        x2 = self.feat_conv2(pre2)
        self.x2 = x2 = self.RDB1(x2)

        if w == None:
            self.fused = x = self.AsyCA1(x1, x2)
        else:
            self.fused = x = w*x1+(1-w)*x2
        # 


        x = self.RDB2(x)
        x = self.out_conv(x)
        # x = self.tanh(x)
        return x

