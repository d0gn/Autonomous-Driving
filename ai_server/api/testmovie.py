import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import models.derainhaze as derainhaze
# ================================================
# (B) TorchScript 혹은 .pth 로드 로직 수정
# ================================================
def load_model(model_path, device):
    """
    모델 파일이 TorchScript(.pt)인지 .pth 루 state_dict인지 자동으로 판단하여 로드
    """
    print(f"[Load] 모델 불러오는 중: {model_path}")
    # TorchScript 아카이브인지 시도해보고 실패 시 state_dict 로드
    try:
        # TorchScript 로드 시도
        model_ts = torch.jit.load(model_path, map_location=device)
        print("[Load] TorchScript 모델로 성공적으로 로드됨.")
        model_ts.eval()
        return model_ts

    except RuntimeError as e:
        print("[Load] TorchScript 로드 실패, state_dict로 로드 시도:", e)
        # state_dict면 DerainNet 인스턴스 생성 후 load_state_dict
        state = torch.load(model_path, map_location=device)
        model = derainhaze.DerainNet().to(device)
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)
        model.eval()
        print("[Load] state_dict 방식으로 모델 로드 완료.")
        return model


# ================================================
# (C) 비디오 단위 추론 함수
# ================================================
def derain_video(input_video_path, output_video_path, model, device, target_size=(512, 1024)):
    """
    1) input_video_path: 입력 비디오 파일 경로 (예: .mp4, .avi 등)
    2) output_video_path: 결과를 저장할 출력 비디오 경로
    3) model: 추론에 사용할 PyTorch 모델 (TorchScript 혹은 nn.Module)
    4) device: torch.device("cuda") 또는 ("cpu")
    5) target_size: (height, width) 형태로 모델 입출력 크기, 예: (512, 1024)
    """
    print(f"[Video] 입력 비디오: {input_video_path}")
    print(f"[Video] 출력 비디오: {output_video_path}")
    print(f"[Video] 목표 해상도: {target_size}\n")

    # 0) 비디오 캡처 열기
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"입력 비디오를 열 수 없습니다: {input_video_path}")

    # 원본 비디오 속성 가져오기
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[Video] 원본 해상도: ({orig_height}, {orig_width}), FPS: {orig_fps}, 총 프레임: {frame_count}")

    # 1) 출력 비디오 설정 (FourCC 코덱: MJPG)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(
        output_video_path,
        fourcc,
        orig_fps,
        (orig_width, orig_height)  # 출력 해상도를 원본과 동일하게
    )

    # 2) 프레임 단위 반복
    frame_idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_idx += 1

        # 진행률 출력
        if frame_idx % 30 == 0 or frame_idx == frame_count:
            print(f"[Video] 처리 중... ({frame_idx}/{frame_count})")

        # a) BGR -> RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # b) 모델 입력 크기로 리사이즈
        resized = cv2.resize(frame_rgb, (target_size[1], target_size[0]))  # (W, H)
        # c) 0-1 정규화 및 Tensor 변환
        inp = resized.astype(np.float32) / 255.0                          # (H, W, 3), float32
        inp = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0).to(device)  # (1,3,H,W)

        # d) 모델 추론
        with torch.no_grad():
            out_tensor = model(inp)  # (1,3,H,W)
        # e) 후처리: (1,3,H,W)->(H,W,3)->uint8->BGR
        out_np = out_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        out_np = (out_np * 255.0).clip(0, 255).astype(np.uint8)
        out_bgr = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)
        # f) 원본 해상도로 다시 리사이즈
        out_frame = cv2.resize(out_bgr, (orig_width, orig_height))

        # g) 출력 비디오에 쓰기
        out.write(out_frame)

    # 3) 해제
    cap.release()
    out.release()
    print("\n[Video] 처리 완료! 출력 비디오 저장됨:", output_video_path)


# ================================================
# (D) 메인 실행부
# ================================================
if __name__ == "__main__":
    print("===== 영상 단위 추론 스크립트 시작 =====")

    # 1) 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Config] Device: {device}")

    # 2) 모델 파일 경로 (TorchScript .pt 또는 .pth)
    model_path = "epoch60+datasetplus.pt"  # 실제 경로로 수정
    model = load_model(model_path, device)

    # 3) 입력/출력 비디오 경로 설정
    input_video_path = "rain.mp4"      # 예: 입력 비디오 파일
    output_video_path = "output_derained_video.avi" # 출력 비디오 파일 (MJPG 코덱 사용)

    # 4) 모델이 학습된 해상도로 지정 (height, width)
    target_size = (500, 500)  # 학습 시 사용한 크기와 동일해야 함

    # 5) 비디오 추론 수행
    derain_video(input_video_path, output_video_path, model, device, target_size)

    print("===== 영상 단위 추론 스크립트 종료 =====")
