"""
CaLLE is implemented based on https://github.com/Thehunk1206/Zero-DCE
"""
import torch
import torch.nn as nn
from Model.Modules import ConvBNGeLU, SqueezeExcitation, ConvBN

class CSDN_Tem(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CSDN_Tem, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


class BasicConv(nn.Module):
    def __init__(self, inp, oup, stride=1, expand_ratio=4, dilation=(1,2,3), residual=True):
        super(BasicConv, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        if self.stride == 1 and inp == oup:
            self.use_res_connect = residual
        else:
            self.use_res_connect = False

        self.conv1 = ConvBNGeLU(inp, hidden_dim, kernel_size=1)

        self.hidden_conv1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=dilation[0], groups=hidden_dim, dilation=dilation[0])
        self.hidden_conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=dilation[1], groups=hidden_dim, dilation=dilation[1])
        self.hidden_conv3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=dilation[2], groups=hidden_dim, dilation=dilation[2])
        self.hidden_bnact = nn.Sequential(nn.BatchNorm2d(hidden_dim), nn.GELU())
        self.out_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        m = self.conv1(x)
        m = self.hidden_conv1(m) + self.hidden_conv2(m) + self.hidden_conv3(m)
        m = self.hidden_bnact(m)
        if self.use_res_connect:
            return x + self.out_conv(m)
        else:
            return self.out_conv(m)


class COD_aware_Distraction(nn.Module):
    def __init__(self, channel):
        super(COD_aware_Distraction, self).__init__()

        # pre_conv replacing to multi-scale module
        self.pre_conv = BasicConv(channel, channel)
        self.map_conv = nn.Conv2d(channel, 1, kernel_size=3, stride=1, padding=1, bias=True)

        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))

        self.f_conv1 = BasicConv(channel, channel)
        self.f_conv2 = ConvBN(channel, channel, kernel_size=3)

        self.b_conv1 = BasicConv(channel, channel)
        self.b_conv2 = ConvBN(channel, channel, kernel_size=3)

        self.gelu = nn.GELU()
        self.out_conv1 = ConvBNGeLU(channel, channel//2, kernel_size=3)
        self.out_conv2 = nn.Conv2d(channel//2, 1, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        x = self.pre_conv(x)
        map = torch.sigmoid(self.map_conv(x))
        f_f = x * map
        b_f = x * (1.0 - map)

        f_f = self.f_conv2(self.f_conv1(f_f) * self.alpha)
        b_f = self.b_conv2(self.b_conv1(b_f) * self.beta)

        f = self.out_conv1(self.gelu(f_f + b_f))
        f = self.out_conv2(f)
        return f


class CaLLE(nn.Module):

    def __init__(self, channel=16):
        super(CaLLE, self).__init__()

        self.gelu = nn.GELU()

        #   zerodce DWC + p-shared
        self.e_conv1 = CSDN_Tem(3, channel)
        self.e_conv2 = CSDN_Tem(channel, channel)
        self.e_conv3 = CSDN_Tem(channel, channel)
        self.e_conv4 = CSDN_Tem(channel, channel)
        self.e_conv5 = CSDN_Tem(channel * 2, channel)
        self.e_conv6 = CSDN_Tem(channel * 2, channel)
        self.e_conv7 = CSDN_Tem(channel * 2, 3)

        # SE
        self.se1 = SqueezeExcitation(in_channels=channel, out_channels=channel)
        self.se2 = SqueezeExcitation(in_channels=channel, out_channels=channel)
        self.se3 = SqueezeExcitation(in_channels=channel, out_channels=channel)
        self.se4 = SqueezeExcitation(in_channels=channel, out_channels=channel)

        # COD-aware Distraction
        self.codAD = COD_aware_Distraction(channel)

        # high-level guidance
        self.hg3 = nn.Conv2d(channel, 1, kernel_size=3, stride=1, padding=1, bias=True)
        self.hg2 = nn.Conv2d(channel, 1, kernel_size=3, stride=1, padding=1, bias=True)
        self.hg1 = nn.Conv2d(channel, 1, kernel_size=3, stride=1, padding=1, bias=True)

        # init weights
        self.init_weights()

    def enhance(self, x, x_r):
        x = x + x_r * (torch.pow(x, 2) - x)
        x = x + x_r * (torch.pow(x, 2) - x)
        x = x + x_r * (torch.pow(x, 2) - x)
        enhance_image_1 = x + x_r * (torch.pow(x, 2) - x)
        x = enhance_image_1 + x_r * (torch.pow(enhance_image_1, 2) - enhance_image_1)
        x = x + x_r * (torch.pow(x, 2) - x)
        x = x + x_r * (torch.pow(x, 2) - x)
        enhance_image = x + x_r * (torch.pow(x, 2) - x)

        return enhance_image

    def forward(self, x):
        x1 = self.gelu(self.e_conv1(x))
        x2 = self.gelu(self.e_conv2(x1))
        x3 = self.gelu(self.e_conv3(x2))
        x4 = self.gelu(self.e_conv4(x3))
        x_cod = self.codAD(x4)
        map_cod = torch.sigmoid(x_cod)
        x4 = self.se4(x4 * map_cod)
        x4_g = torch.sigmoid(self.hg3(x4))
        x3 = self.se3(x3 * map_cod) * x4_g
        x5 = self.gelu(self.e_conv5(torch.cat([x3, x4], 1)))
        x5_g = torch.sigmoid(self.hg2(x5))
        x2 = self.se2(x2 * map_cod) * x5_g
        x6 = self.gelu(self.e_conv6(torch.cat([x2, x5], 1)))
        x6_g = torch.sigmoid(self.hg1(x6))
        x1 = self.se1(x1 * map_cod) * x6_g
        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        enhance_image = self.enhance(x, x_r)
        return enhance_image, x_r, x_cod

    def init_weights(self):
        for name, param in self.named_parameters():
            if name.find('Conv') != -1:
                param.weight.data.normal_(0.0, 0.02)
            elif name.find('BatchNorm') != -1:
                param.weight.data.normal_(1.0, 0.02)
                param.bias.data.fill_(0)

