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
import models.derainhaze as derainhaze
import prepros
import VGGPerceptual




# ================================================
# 3) 학습 예시: DataLoader + 학습 루프 (Windows 멀티프로세싱 안전 진입점)
# ================================================
if __name__ == "__main__":
    print("===== 학습 스크립트 시작 =====")

    # 1) 사용자 설정: 경로 및 해상도
    root_gt_folder = "C:/Users/iliad/Downloads/RainDS/RainDS/RainDS_syn/train/gt"
    root_rain_folder = "C:/Users/iliad/Downloads/RainDS/RainDS/RainDS_syn/train/rainstreak_raindrop"
    img_h, img_w = 500,500
    batch_size       = 8
    val_batch_size   = 4
    num_epochs       = 50
    lr               = 1e-3
    print(f"[Config] GT 폴더: {root_gt_folder}")
    print(f"[Config] Rain 폴더: {root_rain_folder}")
    print(f"[Config] 이미지 크기: ({img_h}, {img_w})\n")

    # 2) 데이터셋 및 DataLoader 생성
    print("[Main] 데이터셋 생성 중...")
    dataset = prepros.RainDSSynDataset(
        root_gt=root_gt_folder,
        root_rain=root_rain_folder,
        img_size=(img_h, img_w),
        transform=None  # 기본 Resize + ToTensor 사용
    )
    print("[Main] DataLoader 설정 중...")
    loader = DataLoader(
        dataset,
        batch_size=8,     # 배치 크기
        shuffle=True,
        num_workers=1,    # 워커 수 (Windows에서는 0 또는 1로 줄여서 테스트 권장)
        pin_memory=True
    )
    print(f"[Main] 데이터로더 크기: {len(loader)} 배치\n")
    val_gt_folder   = "C:/Users/iliad/Downloads/RainDS/RainDS/RainDS_syn/test/gt"
    val_rain_folder = "C:/Users/iliad/Downloads/RainDS/RainDS/RainDS_syn/test/rainstreak_raindrop"
    val_dataset = prepros.RainDSSynDataset(
        root_gt=val_gt_folder,
        root_rain=val_rain_folder,
        img_size=(img_h, img_w),
        transform=transforms.Compose([
            transforms.Resize((img_h, img_w)),
            transforms.ToTensor(),
        ])
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    # 3) 장치(device) 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Main] Using device: {device}\n")

    # 4) 모델, 옵티마이저, 손실함수 정의
    print("[Main] 모델 초기화 중...")
    model = derainhaze.DerainNet().to(device)
    print("[Main] 옵티마이저 및 손실 함수 설정 중...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    criterion = nn.MSELoss()
    print("[Main] 준비 완료\n")

    # 5) 에폭 수 정의
    num_epochs = 50  # 원하는 에폭 수로 설정하세요
    print(f"[Main] 학습 에폭 수: {num_epochs}\n")
    # Perceptual Loss용 VGG 추출기 생성
    vgg_extractor = VGGPerceptual(requires_grad=False).to(device)

    # λ (Perceptual Loss 가중치) 설정
    lambda_perc = 0.01
    # 6) 학습 루프
    for epoch in range(num_epochs):
        print(f"[Epoch {epoch+1}/{num_epochs}] 학습 시작")
        model.train()
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(loader, start=1):
            rain_imgs = batch["rain"].to(device)  # (B,3,H,W)
            gt_imgs   = batch["gt"].to(device)    # (B,3,H,W)

            optimizer.zero_grad()
            outputs = model(rain_imgs)            # (B,3,H,W)
             # 1) 픽셀 MSE 손실
            loss_mse = criterion(outputs, gt_imgs)
            # 2) Perceptual Loss
            loss_perc = VGGPerceptual.perceptual_loss(outputs, gt_imgs, vgg_extractor)
            # 3) 총 손실 = MSE + λ * Perceptual
            loss = loss_mse + lambda_perc * loss_perc

            #loss = criterion(outputs, gt_imgs)    # 단순 MSE 손실
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            epoch_loss += batch_loss * rain_imgs.size(0)

            # 배치 단위 진행 상황 출력
            if batch_idx % 10 == 0 or batch_idx == len(loader):
                print(f"  [Batch {batch_idx}/{len(loader)}]  Loss: {batch_loss:.6f}")

        epoch_loss /= len(dataset)
        print(f"[Epoch {epoch+1}/{num_epochs}]  평균 Loss: {epoch_loss:.6f}\n")
    
    # 7) 학습 완료 후 TorchScript로 저장 (추론용)
    print("[Main] 학습 완료, TorchScript 모델로 변환 중...")
    model.eval()
    example = torch.randn(1, 3, img_h, img_w).to("cpu")
    traced = torch.jit.trace(model.cpu(), example)
    traced.save("derain_dehaze_rainDS_syn_ts.pt")
    print("[Main] TorchScript 모델 저장 완료: derain_dehaze_rainDS_syn_ts.pt")
    print("===== 학습 스크립트 종료 =====")
