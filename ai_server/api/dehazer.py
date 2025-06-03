#디헤이징 테스트코드 삭제요망
import torch
import torchvision
import numpy as np
import cv2
from PIL import Image 
from torchvision import transforms 
import os
import sys
from pathlib import Path

script_dir = Path(Path(__file__).parent).parent
model_dir = script_dir / 'models' 
if not model_dir.exists():
    print(f"디렉토리 존재 x: {model_dir}")
else:
    sys.path.append(str(model_dir))
    print(f"'{model_dir}' 경로추가")

import net
import time

def apply_dehazing(image_np_bgr, dehaze_model, device):
    if dehaze_model is None:
        print("디헤이징 모델 로딩안됨")
        return image_np_bgr 

    if image_np_bgr is None or image_np_bgr.size == 0:
        print("이미지가 유효하지 않음 ")
        return image_np_bgr 

    processed_image_np_bgr = image_np_bgr

    try:
        print("디헤이징 처리 ")
        image_np_rgb = cv2.cvtColor(image_np_bgr, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(image_np_rgb.copy()).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        image_tensor = image_tensor.to(device)

        # 디헤이징 모델 적용
        with torch.no_grad(): # 추론 시에는 그래디언트 계산 비활성화
            # 모델 출력은 일반적으로 [0, 1] 범위의 RGB Tensor (NCHW)
            dehazed_tensor = dehaze_model(image_tensor)

        # 디헤이징 결과 Tensor를 OpenCV BGR numpy 배열로 변환
        # 배치 차원 제거, CxHxW -> HxWx3
        dehazed_np_rgb = dehazed_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        # [0, 1] 범위 -> [0, 255] 범위, uint8 타입으로 변환
        dehazed_np_rgb = (dehazed_np_rgb * 255.0).astype(np.uint8)
        # RGB -> BGR
        dehazed_np_bgr = cv2.cvtColor(dehazed_np_rgb, cv2.COLOR_RGB2BGR)

        processed_image_np_bgr = dehazed_np_bgr # 디헤이징 성공 시 결과 이미지 사용

        print("디헤이징 완료료")

        timestamp = int(time.time())
        cv2.imwrite(f"dehazed_output_{timestamp}.jpg", processed_image_np_bgr)
        print(f"디헤이징 이미지 저장 dehazed_output_{timestamp}.jpg")


    except Exception as e:
        print(f"디헤이징 오류발생 {e}")
        # 오류 발생 시 원본 이미지를 그대로 사용
        processed_image_np_bgr = image_np_bgr
        print("디헤이징 실패 ")

    return processed_image_np_bgr

# if __name__ == '__main__':
#    print("실험")
#    pass

