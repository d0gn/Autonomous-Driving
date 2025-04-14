import torch
import torch as nn
import torch.nn.functional as F

class meodnet(nn.module):
    def __init__(self):
        super(meodnet,self).__init()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=7, stride=1, padding=3)
        self.conv5 = nn.Conv2d(in_channels=12, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.b = 1
    def forward(self,x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        cat1 = torch.cat((x1, x2), 1)
        x3 = F.relu(self.conv3(cat1))
        cat2 = torch.cat((x2, x3), 1)
        x4 = F.relu(self.conv4(cat2))
        cat3 = torch.cat((x1, x2, x3, x4), 1)
        k = F.relu(self.conv5(cat3))

        if k.size() != x.size():
            raise Exception("size is different")

        output = k * x - k + self.b
        return F.relu(output)


class AODnet(nn.Module):
    def __init__(self):
        super(AODnet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=7, stride=1, padding=3)
        self.conv5 = nn.Conv2d(in_channels=12, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.b = 1

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        cat1 = torch.cat((x1, x2), 1)
        x3 = F.relu(self.conv3(cat1))
        cat2 = torch.cat((x2, x3), 1)
        x4 = F.relu(self.conv4(cat2))
        cat3 = torch.cat((x1, x2, x3, x4), 1)
        k = F.relu(self.conv5(cat3))

        if k.size() != x.size():
            raise Exception("k, haze image are different size!")

        output = k * x - k + self.b
        return F.relu(output)



class RainStreakRemovalNet(nn.Module):
    def __init__(self):
        super(RainStreakRemovalNet, self).__init__()

        # K1 Branch
        self.k1_conv1 = nn.Conv2d(3, 16, kernel_size=1, stride=1, padding=1)
        self.k1_conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.k1_conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.k1_conv4 = nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1)

        # K2 Branch
        self.k2_conv1 = nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=1)
        self.k2_conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.k2_conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.k2_conv4 = nn.Conv2d(32, 96, kernel_size=3, stride=1, padding=1)

        # Final conv layers
        self.final_conv1 = nn.Conv2d(96, 16, kernel_size=1, stride=1, padding=0)
        self.final_conv2 = nn.Conv2d(16, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # K1 Branch
        k1_1 = self.k1_conv1(x)
        k1_2 = self.k1_conv2(k1_1)
        k1_3 = self.k1_conv3(k1_2)
        k1_4 = self.k1_conv4(k1_3)

        # K2 Branch
        k2_1 = self.k2_conv1(x)
        k2_2 = self.k2_conv2(k2_1)
        k2_3 = self.k2_conv3(k2_2)
        k2_4 = self.k2_conv4(k2_3)

        # Concatenate K1 outputs
        k1_concat = torch.cat((k1_1, k1_2, k1_3, k1_4), dim=1)

        # Concatenate K2 outputs
        k2_concat = torch.cat((k2_1, k2_2, k2_3, k2_4), dim=1)

        # Combine branches
        combined = k1_concat - k2_concat

        # Final convolutions
        out = self.final_conv1(combined)
        out = self.final_conv2(out)

        return out

# 모델 인스턴스 생성
model = RainStreakRemovalNet()

# 임의의 입력 텐서 (배치 크기, 채널 수, 높이, 너비)
input_tensor = torch.randn(1, 3, 256, 256)
output = model(input_tensor)

print(output.shape)  # 출력 형태 확인