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

class VGGPerceptual(nn.Module):
    """
    VGG16 기반으로 relu2_2 레이어까지 특징 맵을 추출합니다.
    사전 학습된(pretrained) 파라미터는 모두 freeze 처리합니다.
    """
    def __init__(self, requires_grad=False):
        super(VGGPerceptual, self).__init__()
        vgg_pretrained = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        # relu2_2까지(인덱스 0~8) 가져오기
        self.slice = nn.Sequential(*[vgg_pretrained[i] for i in range(9)])
        if not requires_grad:
            for param in self.slice.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.slice(x)


# ----------------------------
# 2) 정규화 함수 (ImageNet 기준)
# ----------------------------
def normalize_batch(x: torch.Tensor) -> torch.Tensor:
    """
    x: (B, 3, H, W), 값 범위 [0,1]
    ImageNet 평균/표준편차로 정규화하여 반환
    """
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


# ----------------------------
# 3) Perceptual Loss 함수
# ----------------------------
def perceptual_loss(prediction: torch.Tensor, 
                    target: torch.Tensor, 
                    vgg_extractor: nn.Module) -> torch.Tensor:
    """
    prediction, target: (B, 3, H, W), 값 범위 [0,1]
    vgg_extractor: VGGPerceptual 인스턴스
    """
    # 1) ImageNet 정규화
    pred_norm = normalize_batch(prediction)
    tgt_norm  = normalize_batch(target)
    
    # 2) VGG 특징 맵 추출 (relu2_2 출력)
    pred_feat = vgg_extractor(pred_norm)
    tgt_feat  = vgg_extractor(tgt_norm)
    
    # 3) MSE로 비교
    return F.mse_loss(pred_feat, tgt_feat)
def compute_batch_psnr_ssim(
                                preds: torch.Tensor,
                                targets: torch.Tensor,
                                data_range: float = 1.0
                            ) -> Tuple[float, float]:
    """
    한 배치(B) 내 PSNR/SSIM 평균 계산
    preds, targets: (B, C, H, W), 값 [0,1]
    """
    preds_np = preds.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    batch_size = preds_np.shape[0]
    psnr_sum = 0.0
    ssim_sum = 0.0
    
    for i in range(batch_size):
        pred_img = np.transpose(preds_np[i], (1, 2, 0))   # (H, W, C)
        target_img = np.transpose(targets_np[i], (1, 2, 0))  # (H, W, C)
        
        psnr_sum += peak_signal_noise_ratio(target_img, pred_img, data_range=data_range)
        ssim_sum += structural_similarity(target_img, pred_img, 
                                          data_range=data_range, multichannel=True)
    return psnr_sum / batch_size, ssim_sum / batch_size