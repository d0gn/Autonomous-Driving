import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import cv2
import numpy as np
import models.derainhaze as derainhaze

# ================================================
# (B) 학습된 TorchScript (.pt) 또는 일반 state_dict (.pth) 파일 불러오기
# ================================================
def load_trained_model(path, device):
    """
    - 파일 확장자가 '.pt'이면 torch.jit.load()를 시도
    - 그렇지 않으면 torch.load()로 state_dict를 불러와 직접 로드
    """
    extension = os.path.splitext(path)[1].lower()
    if extension == ".pt":
        print(f"[Load] TorchScript 아카이브 '{path}' 로드 중...")
        model = torch.jit.load(path, map_location=device)
        model.to(device)
        model.eval()
        print("[Load] TorchScript 모델 로드 완료 (eval 모드).\n")
        return model

    else:
        print(f"[Load] state_dict 아카이브 '{path}' 로드 중...")
        model = derainhaze.DerainNet().to(device)
        checkpoint = torch.load(path, map_location=device)

        # checkpoint가 딕셔너리라면, 내부에 'model_state_dict' 키가 있는지 확인
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)
        model.eval()
        print("[Load] state_dict 모델 로드 완료 (eval 모드).\n")
        return model


# ================================================
# 3) 단일 이미지 추론 예시
# ================================================
if __name__ == "__main__":
    print("===== 추론 스크립트 시작 =====")

    # (1) device 설정 (GPU가 없으면 CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Config] Using device: {device}")

    # (2) 학습된 파일 경로 설정 (.pt 또는 .pth)
    #     예: "derain_dehaze_rainDS_syn_ts.pt"  혹은  "derain_dehaze_rainDS_syn.pth"
    trained_path = "epoch60+datasetplus.pt"
    model = load_trained_model(trained_path, device)

    # (3) 추론할 이미지 경로 지정
    sample_rain_img = "./asd.jpg"
    print(f"[Inference] 처리할 이미지: {sample_rain_img}")

    # (4) 이미지 열기 및 전처리
    img_bgr = cv2.imread(sample_rain_img)
    if img_bgr is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {sample_rain_img}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 학습 때 사용한 해상도로 Resize (예: 512×1024)
    H, W = 512, 512
    img_resized = cv2.resize(img_rgb, (W, H))   # (W, H) 순서
    img_f = img_resized.astype(np.float32) / 255.0

    # (H, W, C) → (C, H, W) → (1, C, H, W)
    input_tensor = torch.from_numpy(img_f).permute(2, 0, 1).unsqueeze(0).to(device)
    print(f"[Inference] 입력 텐서 크기: {input_tensor.shape}")

    # (5) 모델 추론
    with torch.no_grad():
        output_tensor = model(input_tensor)  # (1, 3, H, W)
    print(f"[Inference] 출력 텐서 크기: {output_tensor.shape}")

    # (6) 후처리: (1, 3, H, W) → (H, W, 3) → 0~255 uint8 → BGR로 변환 → 파일 저장
    out_np = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
    out_np = (out_np * 255.0).clip(0, 255).astype(np.uint8)
    out_bgr = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)

    save_path = "dd.png"
    cv2.imwrite(save_path, out_bgr)
    print(f"[Inference] 복원된 이미지 저장 완료: {save_path}")

    # (7) 폴더 단위 추론 예시 (필요 시 주석 해제 후 사용)
    # rain_folder = "C:/Users/iliad/Downloads/RainDS/RainDS/RainDS_syn/train/rainstreak_raindrop"
    # save_folder = "C:/Users/iliad/Downloads/RainDS/RainDS/InferenceResults"
    # os.makedirs(save_folder, exist_ok=True)
    # for rain_fname in os.listdir(rain_folder):
    #     if not rain_fname.lower().endswith((".png", ".jpg", ".jpeg")):
    #         continue
    #     full_rain_path = os.path.join(rain_folder, rain_fname)
    #     img_bgr = cv2.imread(full_rain_path)
    #     img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    #     img_resized = cv2.resize(img_rgb, (W, H))
    #     img_f = img_resized.astype(np.float32) / 255.0
    #     inp = torch.from_numpy(img_f).permute(2, 0, 1).unsqueeze(0).to(device)
    #
    #     with torch.no_grad():
    #         out_t = model(inp)
    #     out_np2 = out_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
    #     out_np2 = (out_np2 * 255.0).clip(0, 255).astype(np.uint8)
    #     out_bgr2 = cv2.cvtColor(out_np2, cv2.COLOR_RGB2BGR)
    #
    #     save_fname = os.path.splitext(rain_fname)[0] + "_derained.png"
    #     cv2.imwrite(os.path.join(save_folder, save_fname), out_bgr2)
    #
    # print("[Inference] 폴더 단위 처리 완료")

    print("===== 추론 스크립트 종료 =====")
