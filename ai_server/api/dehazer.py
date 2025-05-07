import torch
import torchvision
import numpy as np
import cv2
from PIL import Image # 이미지 변환 시 필요할 수 있음
from torchvision import transforms # 이미지 전처리 시 필요

# net 모듈 임포트 (dehaze_net 아키텍처 정의가 필요함)
# 이 파일(dehazer_module.py)이 net.py와 같은 디렉토리나 sys.path에 있는 곳에 있어야 합니다.
import net


def apply_dehazing(image_np_bgr, dehaze_model, device):
    """
    OpenCV 이미지 (BGR, numpy 배열)에 디헤이징 모델을 적용합니다.

    Args:
        image_np_bgr (numpy.ndarray): OpenCV로 읽은 이미지 데이터 (BGR 형식, uint8)
        dehaze_model (torch.nn.Module): 미리 로딩된 Dehazing 모델 인스턴스
        device (torch.device): 모델 추론에 사용할 장치 (cuda 또는 cpu)

    Returns:
        numpy.ndarray: 디헤이징 처리된 이미지 데이터 (BGR 형식, uint8),
                       오류 발생 시 원본 이미지 반환
    """
    if dehaze_model is None:
        print("🚨 Dehazing 모델이 로딩되지 않았습니다. 디헤이징을 적용하지 않습니다.")
        return image_np_bgr # 모델이 없으면 원본 이미지 반환

    if image_np_bgr is None or image_np_bgr.size == 0:
        print("🚨 apply_dehazing: 유효하지 않은 입력 이미지입니다.")
        return image_np_bgr # 유효하지 않은 이미지 입력 시 원본 이미지 반환

    processed_image_np_bgr = image_np_bgr # 기본적으로 원본 이미지 사용

    try:
        print("✨ apply_dehazing: 이미지 디헤이징 처리 중...")
        # OpenCV BGR (HxWx3) numpy 배열을 PyTorch 입력 형식으로 변환
        # BGR -> RGB
        image_np_rgb = cv2.cvtColor(image_np_bgr, cv2.COLOR_BGR2RGB)
        # HxWx3 numpy -> CxHxW tensor, 정규화 [0, 1]
        image_tensor = torch.from_numpy(image_np_rgb.copy()).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        image_tensor = image_tensor.to(device) # 모델이 있는 장치로 이동

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

        print("✅ apply_dehazing: 디헤이징 완료.")

        # 디버깅을 위해 디헤이징된 이미지를 파일로 저장할 수 있습니다.
        # timestamp = int(time.time())
        # cv2.imwrite(f"dehazed_output_{timestamp}.jpg", processed_image_np_bgr)
        # print(f"apply_dehazing: 디헤이징된 이미지 임시 저장됨: dehazed_output_{timestamp}.jpg")


    except Exception as e:
        print(f"❌ apply_dehazing: 디헤이징 처리 중 오류 발생: {e}")
        # 오류 발생 시 원본 이미지를 그대로 사용
        processed_image_np_bgr = image_np_bgr
        print("apply_dehazing: 디헤이징 실패, 원본 이미지 반환.")

    return processed_image_np_bgr

# 이 파일 자체를 직접 실행할 경우 테스트 코드 등을 추가할 수 있습니다.
# if __name__ == '__main__':
#    print("dehazer_module.py 파일을 직접 실행했습니다. 보통은 다른 스크립트에서 임포트됩니다.")
#    # 테스트 코드 예시: 이미지 로드, 디헤이징 함수 호출 등
#    pass

