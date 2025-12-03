import os
import sys
import time
from pathlib import Path
import base64 

import socketio 
import eventlet 
import eventlet.wsgi 

import cv2 
import numpy as np 
from PIL import Image 

import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
from torchvision import transforms 

# --- 경로 설정 ---
script_dir = Path(__file__).parent

# 1. models 폴더 경로 추가 (derainhaze.py를 찾기 위해)
models_dir = script_dir / 'models'
if models_dir.exists():
    sys.path.append(str(models_dir))
    print(f"Path added: {models_dir}")
else:
    print(f"Warning: Models dir not found at {models_dir}")

# 2. api 폴더 경로 추가 (yolodetect 등을 위해 유지)
api_dir = script_dir / 'api'
if api_dir.exists():
    sys.path.append(str(api_dir))

# --- 모듈 임포트 ---

# [수정] 기존 net, dehazer 모듈 제거하고 derainhaze만 임포트
try:
    import models.derainhaze as TBAodNet
    print("성공: derainhaze 모듈 임포트")
except ImportError:
    print("실패: derainhaze 모듈을 찾을 수 없음 (models 폴더 확인 필요)")
    derainhaze = None

# yolo 임포트
try:
    import yolodetect as yd 
    print("성공: yolodetect 모듈 임포트")
except ImportError:
    print("실패: yolodetect 모듈 임포트")
    yd = None


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

global_dehaze_net = None
global_yolo_detector = None


# --- [새로 구현] 디헤이징 모델 적용 함수 (Inference Logic) ---
# 기존 dehazer.apply_dehazing을 대체하여 직접 구현
def run_dehazing_inference(image_bgr, model, device):
    """
    이미지(BGR numpy)를 받아 모델 전처리 -> 추론 -> 후처리 후 깨끗한 이미지(BGR numpy) 반환
    """
    if model is None:
        return image_bgr

    try:
        # 1. Preprocess: BGR -> RGB 및 Tensor 변환
        # OpenCV 이미지를 PIL로 변환하거나 바로 Tensor로 변환
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # 정규화 (0~1 사이 값으로 변환) 및 차원 변경 (HWC -> BCHW)
        # 일반적인 PyTorch 모델 입력: (Batch, Channel, Height, Width)
        img_tensor = transforms.ToTensor()(img_rgb).unsqueeze(0).to(device)

        # 2. Inference: 모델 적용
        with torch.no_grad():
            clean_tensor = model(img_tensor)

        # 3. Postprocess: Tensor -> Numpy BGR
        # 결과가 보통 0~1 사이이므로 clamp 후 255 곱하기
        clean_tensor = torch.clamp(clean_tensor, 0, 1)
        
        # GPU Tensor -> CPU -> Numpy 변환
        clean_numpy = clean_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() # CHW -> HWC
        
        # 0~1 float -> 0~255 uint8 변환
        clean_image_uint8 = (clean_numpy * 255).astype('uint8')
        
        # RGB -> BGR 변환 (OpenCV용)
        clean_bgr = cv2.cvtColor(clean_image_uint8, cv2.COLOR_RGB2BGR)
        
        return clean_bgr

    except Exception as e:
        print(f"디헤이징 적용 중 오류 발생: {e}")
        return image_bgr # 오류 시 원본 반환


# --- 모델 로딩 함수 ---
def load_models():
    global global_dehaze_net, global_yolo_detector, DEVICE

    print("--- 모델 로딩 시작 ---")

    # 1. 디헤이징 모델 (TBAodNet) 로딩
    print("1. Dehazing Model Loading...")
    if derainhaze is None:
        print(" -> derainhaze 모듈이 없어 로딩 건너뜀")
    else:
        try:
            checkpoint_path = script_dir / 'pt' / 'dehazer.pth'

            print(f" -> 체크포인트 경로: {checkpoint_path}")

            if not checkpoint_path.exists():
                print(" -> 실패: 체크포인트 파일이 존재하지 않음")
                global_dehaze_net = None
            else:
                global_dehaze_net = TBAodNet()
                
                # 가중치 로드
                global_dehaze_net.load_state_dict(torch.load(str(checkpoint_path), map_location=DEVICE))
                global_dehaze_net.to(DEVICE)
                global_dehaze_net.eval()
                print(" -> 성공: TBAodNet 로딩 완료")
                
        except Exception as e:
            print(f" -> 실패: 디헤이징 모델 로딩 중 에러 ({e})")
            global_dehaze_net = None

    # 2. YOLO 모델 로딩
    print("2. YOLO Model Loading...")
    if yd is None:
         print(" -> yolodetect 모듈 없음")
    else:
        try:
            global_yolo_detector = yd.YOLODetector(weights_path='yolov5s.pt', device=str(DEVICE), img_size=640) 

            if global_yolo_detector.model is None:
                 print(" -> 실패: YOLO 모델 초기화 안됨")
            else:
                print(" -> 성공: YOLO 로딩 완료")
        except Exception as e:
            print(f" -> 실패: YOLO 로딩 에러 ({e})")
            global_yolo_detector = None

    print("--- 모델 로딩 종료 ---")


# --- python-socketio 서버 설정 ---
sio = socketio.Server(cors_allowed_origins="*", ping_interval=5, ping_timeout=10, max_http_buffer_size=100000000)
app = socketio.WSGIApp(sio)
web_sio = socketio.Client()

try:
    web_sio.connect('http://web_server:5000')
except Exception as e:
    print(f"Web Server 연결 실패: {e}")


# --- 이미지 처리 파이프라인 ---
def process_image_and_determine_command(image_np_bgr):
    # print("프레임 처리 시작...") # 로그 너무 많으면 주석 처리
    command = None

    if image_np_bgr is None or image_np_bgr.size == 0:
         return None, None

    # --- 단계 1: 이미지 디헤이징 (직접 구현한 함수 호출) ---
    processed_image_np_bgr = image_np_bgr
    if global_dehaze_net is not None:
         # [수정] dehazer 모듈 대신 직접 만든 함수 사용
         processed_image_np_bgr = run_dehazing_inference(image_np_bgr, global_dehaze_net, DEVICE)
    
    # --- 단계 2: YOLO 객체 검출 ---
    detections = []
    if global_yolo_detector is not None:
        try:
            results, annotated_img = global_yolo_detector.detect_array(processed_image_np_bgr)
            if results is not None:
                 detections = global_yolo_detector.extract_detections(results)
                 
                 # (옵션) 디버깅용 저장
                 # if len(detections) > 0:
                 #    cv2.imwrite(f"debug_{int(time.time())}.jpg", annotated_img)
        except Exception as e:
            print(f"YOLO Error: {e}")
            detections = []


    # --- 단계 3: 명령 결정 로직 ---
    person_detected = False
    car_detected = False

    if detections:
        for det in detections:
            if det['confidence'] > 0.5:
                if det['class_name'] == 'person':
                    person_detected = True
                    command = "stop"
                    break # 사람 발견 시 즉시 중단
                elif det['class_name'] == 'car':
                     car_detected = True

    if command is None and car_detected:
         command = "forward"

    # print(f"결정된 명령: {command}") # 로그 확인용
    return command, processed_image_np_bgr


# --- 소켓 이벤트 핸들러 ---
@sio.on('connect')
def handle_connect(sid, environ):
    print(f'Client Connected (SID: {sid})')

@sio.on('disconnect')
def handle_disconnect(sid):
    print(f'Client Disconnected (SID: {sid})')

@sio.on('image_frame')
def handle_image_frame(sid, data):
    # print(f"이미지 수신 (SID: {sid}) len={len(data)}") 

    try:
        # Base64 디코딩
        image_bytes = base64.b64decode(data)
        npimg = np.frombuffer(image_bytes, np.uint8)
        image_np_bgr = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if image_np_bgr is None:
             return
        
        # 처리 파이프라인 실행
        command_to_send, processed_image = process_image_and_determine_command(image_np_bgr)
        
        # 결과 이미지 인코딩 (전송용)
        _, jpeg_encoded = cv2.imencode('.jpg', processed_image)
        jpeg_base64 = base64.b64encode(jpeg_encoded.tobytes()).decode('utf-8')
        
        # 웹 서버로 결과 전송
        if web_sio.connected:
            web_sio.emit('processed_result', {
                'command': command_to_send,
                'frame': jpeg_base64
            })
        
        # RC카 클라이언트로 명령 전송
        if command_to_send:
            sio.emit('command', {'command': command_to_send}, room=sid)
            print(f"CMD Sent: {command_to_send}")

    except Exception as e:
        print(f"Error processing frame: {e}")
        sio.emit('error', {'message': f'Server error: {e}'}, room=sid)


if __name__ == '__main__':
    # 서버 시작 시 모델 로드
    load_models()

    if global_dehaze_net is None:
        print("주의: 디헤이징 모델이 로드되지 않았습니다. 원본 이미지가 사용됩니다.")

    host = '0.0.0.0'
    port = 5001

    print(f"SocketIO Server running at {host}:{port}")
    eventlet.wsgi.server(eventlet.listen((host, port)), app)