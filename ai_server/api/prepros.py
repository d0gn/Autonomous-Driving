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
# 1) 데이터셋 정의: RainDS_syn 학습용 Dataset 클래스
# ================================================
class RainDSSynDataset(Dataset):
    """
    사용자 데이터셋: GT 폴더에는 세 가지 유형의 이미지가 섞여 있고,
    train/rainstreak_raindrop 폴더에는 대응하는 rain 이미지가 있다.

    - GT 폴더:
        1) pie-norain-<id>.png      (대응: pie-rd-rain-<id>.png)
        2) norain-<id>.png          (대응: re-rain-<id>.png)
        3) NYU_<id>.jpg             (대응: NYU_<id>_<x>_<y>.jpg, 다중 가능)

    - Rain 폴더:
        1) pie-rd-rain-<id>.png
        2) re-rain-<id>.png
        3) NYU_<id>_<x>_<y>.jpg

    각 GT–rain 쌍을 튜플로 저장해, DataLoader가 하나씩 꺼낼 수 있도록 구성합니다.
    """
    def __init__(self,
                 root_gt="C:/Users/iliad/Downloads/RainDS/RainDS/RainDS_syn/train/gt",
                 root_rain="C:/Users/iliad/Downloads/RainDS/RainDS/RainDS_syn/train/rainstreak_raindrop",
                 img_size=(512, 1024),
                 transform=None):
        super().__init__()
        print("[Dataset] 초기화 중...")

        self.root_gt = root_gt
        self.root_rain = root_rain
        self.img_size = img_size

        # 1) GT 폴더에서 모든 이미지 목록을 수집 (png, jpg 모두 포함)
        gt_patterns = ["pie-norain-*.png", "norain-*.png", "NYU_*.jpg"]
        self.gt_paths = []
        for pat in gt_patterns:
            self.gt_paths += sorted(glob(os.path.join(self.root_gt, pat)))
        if len(self.gt_paths) == 0:
            raise RuntimeError(f"[Dataset] GT 이미지가 없습니다: {self.root_gt}")
        print(f"[Dataset] GT 총 이미지 개수: {len(self.gt_paths)}")

        # 2) rain 폴더에 있는 모든 이미지 목록 (확장자 구분 없이)
        self.rain_all = sorted(glob(os.path.join(self.root_rain, "*.*")))
        if len(self.rain_all) == 0:
            raise RuntimeError(f"[Dataset] Rain 이미지가 없습니다: {self.root_rain}")

        # 3) GT–rain 매칭: 목록 순회를 통해 pairs 리스트에 (rain_path, gt_path) 저장
        self.pairs = []
        for gt_path in self.gt_paths:
            fname = os.path.basename(gt_path)
            basename, ext = os.path.splitext(fname)

            # 3-1) pie-norain-<id>.png 형태
            if basename.startswith("pie-norain-"):
                parts = basename.split("-")
                idx_str = parts[-1]
                rain_fname = f"pie-rd-rain-{idx_str}.png"
                rain_path = os.path.join(self.root_rain, rain_fname)
                if not os.path.isfile(rain_path):
                    raise FileNotFoundError(f"[Dataset] 대응 rain 파일을 찾을 수 없습니다: {rain_path}")
                self.pairs.append((rain_path, gt_path))

            # 3-2) norain-<id>.png 형태
            elif basename.startswith("norain-"):
                parts = basename.split("-")
                idx_str = parts[-1]
                rain_fname = f"rd-rain-{idx_str}.png"
                rain_path = os.path.join(self.root_rain, rain_fname)
                if not os.path.isfile(rain_path):
                    raise FileNotFoundError(f"[Dataset] 대응 rain 파일을 찾을 수 없습니다: {rain_path}")
                self.pairs.append((rain_path, gt_path))

            # 3-3) NYU_<id>.jpg 형태
            elif basename.startswith("NYU_"):
                # e.g. basename="NYU_1"
                prefix = basename  # "NYU_1"
                # rain 폴더에서 "NYU_1_*_*.jpg" 패턴에 매칭되는 모든 파일을 찾는다.
                pattern = os.path.join(self.root_rain, f"{prefix}_*_*.*")
                matched = sorted(glob(pattern))
                if len(matched) == 0:
                    raise FileNotFoundError(f"[Dataset] 대응 rain 파일을 찾을 수 없습니다: {pattern}")
                # 매칭된 rain 이미지 하나당 GT를 복제하여 pair 생성
                for rain_path in matched:
                    self.pairs.append((rain_path, gt_path))

            else:
                raise ValueError(f"[Dataset] 알 수 없는 GT 파일 형식: {fname}")

        print(f"[Dataset] 총 매칭 쌍 개수: {len(self.pairs)}")
        print("[Dataset] 초기화 완료\n")

        # 4) transform 설정
        if transform is None:
            print("[Dataset] 기본 transform (Resize → ToTensor) 설정")
            self.transform = transforms.Compose([
                transforms.Resize((self.img_size[0], self.img_size[1])),  # (H, W)
                transforms.ToTensor(),  # [0~255] → [0~1], (C, H, W)
            ])
        else:
            print("[Dataset] 사용자 지정 transform 사용")
            self.transform = transform


    def __len__(self):
        return len(self.pairs)


    def __getitem__(self, idx):
        rain_path, gt_path = self.pairs[idx]

        # PIL로 이미지 열기 (원본 모드 그대로)
        gt_img = Image.open(gt_path).convert("RGB")
        rain_img = Image.open(rain_path).convert("RGB")

        # 동일한 transform 적용
        gt_t = self.transform(gt_img)
        rain_t = self.transform(rain_img)

        return {
            "rain": rain_t,        # torch.FloatTensor (C, H, W)
            "gt": gt_t,            # torch.FloatTensor (C, H, W)
            "rain_path": rain_path,
            "gt_path": gt_path
        }
