import os
from glob import glob
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import numpy as np
from torchvision.models import VGG16_Weights

# ================================================
# 2) 모델 정의: DerainNet (Joint Deraining & Dehazing)
# ================================================
class DerainNet(nn.Module):
    """
    Sun et al. 논문 구조 기반 Joint Deraining & Dehazing 모델
    """
    def __init__(self):
        super(DerainNet, self).__init__()
        print("[Model] DerainNet 초기화 중...")

        # ----- K1 브랜치 -----
        self.k1_conv_d1_1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, dilation=1)
        self.k1_bn_d1_1   = nn.BatchNorm2d(16)
        self.k1_conv_d1_2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, dilation=1)
        self.k1_bn_d1_2   = nn.BatchNorm2d(16)

        self.k1_conv_d2_1 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=2, dilation=2)
        self.k1_bn_d2_1   = nn.BatchNorm2d(16)
        self.k1_conv_d2_2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=2, dilation=2)
        self.k1_bn_d2_2   = nn.BatchNorm2d(16)

        self.k1_conv_d3_1 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=3, dilation=3)
        self.k1_bn_d3_1   = nn.BatchNorm2d(16)
        self.k1_conv_d3_2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=3, dilation=3)
        self.k1_bn_d3_2   = nn.BatchNorm2d(16)

        self.k1_fuse_conv1 = nn.Conv2d(16 * 3, 32, kernel_size=1, stride=1, padding=0)
        self.k1_bn_fuse1   = nn.BatchNorm2d(32)
        self.k1_fuse_conv2 = nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0)
        self.k1_bn_fuse2   = nn.BatchNorm2d(16)

        self.k1_out = nn.Conv2d(16, 3, kernel_size=1, stride=1, padding=0)

        # ----- K2 브랜치 -----
        self.k2_conv_d1_1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, dilation=1)
        self.k2_bn_d1_1   = nn.BatchNorm2d(16)
        self.k2_conv_d1_2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, dilation=1)
        self.k2_bn_d1_2   = nn.BatchNorm2d(16)

        self.k2_conv_d2_1 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=2, dilation=2)
        self.k2_bn_d2_1   = nn.BatchNorm2d(16)
        self.k2_conv_d2_2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=2, dilation=2)
        self.k2_bn_d2_2   = nn.BatchNorm2d(16)

        self.k2_conv_d3_1 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=3, dilation=3)
        self.k2_bn_d3_1   = nn.BatchNorm2d(16)
        self.k2_conv_d3_2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=3, dilation=3)
        self.k2_bn_d3_2   = nn.BatchNorm2d(16)

        self.k2_fuse_conv1 = nn.Conv2d(16 * 3, 32, kernel_size=1, stride=1, padding=0)
        self.k2_bn_fuse1   = nn.BatchNorm2d(32)
        self.k2_fuse_conv2 = nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0)
        self.k2_bn_fuse2   = nn.BatchNorm2d(16)

        self.k2_out = nn.Conv2d(16, 3, kernel_size=1, stride=1, padding=0)

        self.negative_slope = 0.2  # LeakyReLU 기울기 상수

        print("[Model] DerainNet 초기화 완료\n")

    def forward(self, x):
        # ========== K1 브랜치 ==========
        f1 = F.leaky_relu(self.k1_bn_d1_1(self.k1_conv_d1_1(x)), negative_slope=self.negative_slope)
        f1 = F.leaky_relu(self.k1_bn_d1_2(self.k1_conv_d1_2(f1)), negative_slope=self.negative_slope)

        f2 = F.leaky_relu(self.k1_bn_d2_1(self.k1_conv_d2_1(f1)), negative_slope=self.negative_slope)
        f2 = F.leaky_relu(self.k1_bn_d2_2(self.k1_conv_d2_2(f2)), negative_slope=self.negative_slope)

        f3 = F.leaky_relu(self.k1_bn_d3_1(self.k1_conv_d3_1(f2)), negative_slope=self.negative_slope)
        f3 = F.leaky_relu(self.k1_bn_d3_2(self.k1_conv_d3_2(f3)), negative_slope=self.negative_slope)

        fuse1 = torch.cat([f1, f2, f3], dim=1)                      # (B,48,H,W)
        fuse1 = F.relu(self.k1_bn_fuse1(self.k1_fuse_conv1(fuse1))) # (B,32,H,W)
        fuse1 = F.relu(self.k1_bn_fuse2(self.k1_fuse_conv2(fuse1))) # (B,16,H,W)

        K1 = self.k1_out(fuse1)  # (B,3,H,W)

        # ========== K2 브랜치 ==========
        g1 = F.leaky_relu(self.k2_bn_d1_1(self.k2_conv_d1_1(x)), negative_slope=self.negative_slope)
        g1 = F.leaky_relu(self.k2_bn_d1_2(self.k2_conv_d1_2(g1)), negative_slope=self.negative_slope)

        g2 = F.leaky_relu(self.k2_bn_d2_1(self.k2_conv_d2_1(g1)), negative_slope=self.negative_slope)
        g2 = F.leaky_relu(self.k2_bn_d2_2(self.k2_conv_d2_2(g2)), negative_slope=self.negative_slope)

        g3 = F.leaky_relu(self.k2_bn_d3_1(self.k2_conv_d3_1(g2)), negative_slope=self.negative_slope)
        g3 = F.leaky_relu(self.k2_bn_d3_2(self.k2_conv_d3_2(g3)), negative_slope=self.negative_slope)

        fuse2 = torch.cat([g1, g2, g3], dim=1)                      # (B,48,H,W)
        fuse2 = F.relu(self.k2_bn_fuse1(self.k2_fuse_conv1(fuse2))) # (B,32,H,W)
        fuse2 = F.relu(self.k2_bn_fuse2(self.k2_fuse_conv2(fuse2))) # (B,16,H,W)

        K2 = self.k2_out(fuse2)  # (B,3,H,W)

        # ========== 최종 복원 식 ==========
        diff = K1 - K2           # (B,3,H,W)
        clean = diff * x - diff  # (B,3,H,W)

        return clean
