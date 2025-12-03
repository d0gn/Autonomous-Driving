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


class TBAodNet(nn.Module):
    def __init__(self):
        super(TBAodNet, self).__init__()
        print("[Model] Multi-Branch AODNet 초기화 중...")

        # ----- K1 Branch (작은 스케일 특징 담당) -----
        self.k1_conv1 = nn.Conv2d(3, 3, kernel_size=1)
        self.k1_conv2 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.k1_conv3 = nn.Conv2d(6, 3, kernel_size=5, padding=2)

        # ----- K2 Branch (중간 스케일 특징 담당) -----
        self.k2_conv1 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.k2_conv2 = nn.Conv2d(3, 3, kernel_size=5, padding=2)
        self.k2_conv3 = nn.Conv2d(6, 3, kernel_size=7, padding=3)
        
        # ----- K3 Branch (큰 스케일 특징 담당) -----
        self.k3_conv1 = nn.Conv2d(3, 3, kernel_size=5, padding=2)
        self.k3_conv2 = nn.Conv2d(3, 3, kernel_size=7, padding=3)
        self.k3_conv3 = nn.Conv2d(6, 3, kernel_size=9, padding=4)

        # ----- 3개 브랜치의 결과를 융합(Fusion)하는 레이어 -----
        # 각 브랜치에서 3채널 결과가 나오므로 총 9채널을 입력받아 최종 3채널 k로 만듦
        self.fusion_conv = nn.Conv2d(9, 3, kernel_size=3, padding=1)
        
        self.b = 1 # 대기 산란 모델의 상수

        print("[Model] Multi-Branch AODNet 초기화 완료\n")


    def forward(self, x):
        # --- K1 Branch 연산 ---
        x1_1 = F.relu(self.k1_conv1(x))
        x1_2 = F.relu(self.k1_conv2(x1_1))
        k1_cat = torch.cat((x1_1, x1_2), dim=1)
        k1 = F.relu(self.k1_conv3(k1_cat)) # K1 결과

        # --- K2 Branch 연산 ---
        x2_1 = F.relu(self.k2_conv1(x))
        x2_2 = F.relu(self.k2_conv2(x2_1))
        k2_cat = torch.cat((x2_1, x2_2), dim=1)
        k2 = F.relu(self.k2_conv3(k2_cat)) # K2 결과

        # --- K3 Branch 연산 ---
        x3_1 = F.relu(self.k3_conv1(x))
        x3_2 = F.relu(self.k3_conv2(x3_1))
        k3_cat = torch.cat((x3_1, x3_2), dim=1)
        k3 = F.relu(self.k3_conv3(k3_cat)) # K3 결과

        # --- Fusion 단계 ---
        k_fused = torch.cat((k1, k2, k3), dim=1)
        k_final = F.relu(self.fusion_conv(k_fused)) # 최종 K

        # --- AOD-Net 복원 공식 적용 ---
        output = k_final * x - k_final + self.b
        return F.relu(output)
