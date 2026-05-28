import torch
import torch.nn as nn
import torchvision.models as models


class MobileNetV2(nn.Module):
    def __init__(self, in_channel=3):
        super(MobileNetV2, self).__init__()

        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

        self.layer0_0 = model.features[0][0]
        self.layer0_1 = model.features[0][1]
        self.layer0_2 = model.features[0][2]
        self.layer1 = model.features[1]
        self.layer2 = model.features[2]
        self.layer3 = model.features[3]
        self.layer4 = model.features[4]
        self.layer5 = model.features[5]
        self.layer6 = model.features[6]
        self.layer7 = model.features[7]
        self.layer8 = model.features[8]
        self.layer9 = model.features[9]
        self.layer10 = model.features[10]
        self.layer11 = model.features[11]
        self.layer12 = model.features[12]
        self.layer13 = model.features[13]
        self.layer14 = model.features[14]
        self.layer15 = model.features[15]
        self.layer16 = model.features[16]
        self.layer17 = model.features[17]

        if in_channel != 3:
            self.layer0_0 = nn.Conv2d(in_channel, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

    def forward(self, x):
        out1 = self.layer1(self.layer0_2(self.layer0_1(self.layer0_0(x))))
        out3 = self.layer3(self.layer2(out1))
        out6 = self.layer6(self.layer5(self.layer4(out3)))
        out13 = self.layer13(self.layer12(self.layer11(self.layer10(self.layer9(self.layer8(self.layer7(out6)))))))
        out17 = self.layer17(self.layer16(self.layer15(self.layer14(out13))))

        return out1, out3, out6, out13, out17

    @staticmethod
    def get_stage_channels():
        return [16, 24, 32, 96, 320]


class MobileNetV2_V2(nn.Module):
    def __init__(self, in_channel=3):
        super(MobileNetV2_V2, self).__init__()

        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

        self.layer0_0 = model.features[0][0]
        self.layer0_1 = model.features[0][1]
        self.layer0_2 = model.features[0][2]
        self.layer1 = model.features[1]
        self.layer2 = model.features[2]
        self.layer3 = model.features[3]
        self.layer4 = model.features[4]
        self.layer5 = model.features[5]
        self.layer6 = model.features[6]
        self.layer7 = model.features[7]
        self.layer8 = model.features[8]
        self.layer9 = model.features[9]
        self.layer10 = model.features[10]
        self.layer11 = model.features[11]
        self.layer12 = model.features[12]
        self.layer13 = model.features[13]
        self.layer14 = model.features[14]
        self.layer15 = model.features[15]
        self.layer16 = model.features[16]

        if in_channel != 3:
            self.layer0_0 = nn.Conv2d(in_channel, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

    def forward(self, x):
        out1 = self.layer1(self.layer0_2(self.layer0_1(self.layer0_0(x))))
        out3 = self.layer3(self.layer2(out1))
        out6 = self.layer6(self.layer5(self.layer4(out3)))
        out13 = self.layer13(self.layer12(self.layer11(self.layer10(self.layer9(self.layer8(self.layer7(out6)))))))
        out16 = self.layer16(self.layer15(self.layer14(out13)))

        return out1, out3, out6, out13, out16

    @staticmethod
    def get_stage_channels():
        return [16, 24, 32, 96, 160]
