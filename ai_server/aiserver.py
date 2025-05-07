import os
import sys
import time
# import argparse
from pathlib import Path
import base64 # 이미지 데이터를 Base64로 인코딩/디코딩하기 위해 필요

import socketio # python-socketio 라이브러리 임포트
import eventlet # 비동기 웹 서버를 위해 eventlet 임포트
import eventlet.wsgi # WSGI 서버를 위해 eventlet.wsgi 임포트

import cv2 # 이미지 처리 및 디코딩에 필요
import numpy as np # 이미지 데이터를 numpy 배열로 다루기 위해 필요
from PIL import Image # 디헤이징 모듈에서 필요할 수 있음
# import glob


# PyTorch 및 관련 라이브러리
import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
from torchvision import transforms # 디헤이징 모듈에서 필요할 수 있음
# import torch.optim

script_dir = Path(__file__).parent
# net.py, yolodetect.py, dehazer_module.py 파일이 위치한 실제 경로에 맞게 수정이 필요합니다.
model_dir = script_dir / 'api' # 예시: 현재 스크립트와 같은 레벨의 'api' 폴더
if not model_dir.exists():
    print(f"🚨 경고: 모델/모듈 파일이 있을 것으로 예상되는 디렉토리가 존재하지 않습니다: {model_dir}")
    print("sys.path에 추가하지 않습니다. import 오류 발생 시 경로를 확인해주세요.")
else:
    sys.path.append(str(model_dir))
    print(f"✅ '{model_dir}' 경로를 sys.path에 추가했습니다.")

# 필요한 모듈 임포트
try:
    import net
    print("✅ net 모듈 임포트 성공.")
except ImportError:
    print("❌ net 모듈 임포트 실패. net.py 파일이 'api' 폴더에 있는지, 또는 sys.path 설정이 올바른지 확인해주세요.")
    net = None

try:
    import yolodetect as yd # <--- yolodetect.py를 임포트하여 yd로 사용
    print("✅ yolodetect 모듈 임포트 성공.")
except ImportError:
    print("❌ yolodetect 모듈 임포트 실패. yolodetect.py 파일이 'api' 폴더에 있는지, 또는 sys.path 설정이 올바른지 확인해주세요.")
    yd = None

try:
    import dehazer
    print("✅ dehazer_module 모듈 임포트 성공.")
except ImportError:
    print("❌ dehazer_module 모듈 임포트 실패. dehazer_module.py 파일이 'api' 폴더에 있는지, 또는 sys.path 설정이 올바른지 확인해주세요.")
    dehazer_module = None


# --- 글로벌 변수 및 모델 로딩 설정 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"💡 모델 추론 장치: {DEVICE}")

global_dehaze_net = None
global_yolo_detector = None

# --- 모델 로딩 함수 ---
def load_models():
    """서버 시작 시 Dehazing 및 YOLO 모델을 로딩합니다."""
    global global_dehaze_net, global_yolo_detector, DEVICE

    print("⏳ 모델 로딩 시작...")

    # Dehazing 모델 로딩
    print("⏳ Dehazing 모델 로딩 중...")
    if net is None or dehazer is None: 
        print("🚨 net 또는 dehazer_module 임포트 실패로 Dehazing 모델 로딩을 건너뜁니다.")
        global_dehaze_net = None
    else:
        try:
            # Dehazing 체크포인트 파일 경로 조합
            checkpoint_path_relative = './checkpoints/dehazer.pth' # <-- 이 경로가 올바른지 확인해주세요.
            checkpoint_path = script_dir / checkpoint_path_relative

            print(f"💡 Dehazing 체크포인트 파일 경로 확인: {checkpoint_path}")

            if not checkpoint_path.exists():
                 print(f"🚨 경고: Dehazing 체크포인트 파일이 없습니다: {checkpoint_path}")
                 print("Dehazing 모델 로딩을 건너뜁니다. Dehazing 없이 YOLO만 실행됩니다.")
                 global_dehaze_net = None
            else:
                global_dehaze_net = net.dehaze_net()
                global_dehaze_net.load_state_dict(torch.load(str(checkpoint_path), map_location=DEVICE))
                global_dehaze_net.to(DEVICE)
                global_dehaze_net.eval()
                print("✅ Dehazing 모델 로딩 완료.")
        except Exception as e:
             print(f"❌ Dehazing 모델 로딩 실패: {e}")
             global_dehaze_net = None


    # YOLO 모델 로딩
    print("⏳ YOLO 모델 로딩 중...")
    if yd is None:
         print("🚨 yolodetect 모듈을 임포트할 수 없어 YOLO 모델 로딩을 건너뜁니다.")
         global_yolo_detector = None
    else:
        try:
            # YOLODetector 클래스 인스턴스 생성 및 가중치 로딩
            # weights_path='yolov5s.pt'는 torch.hub가 자동으로 다운로드할 수 있습니다.
            # 만약 로컬 특정 경로에 가중치 파일이 있다면, 해당 경로를 지정해야 합니다.
            # 예시: 현재 스크립트와 같은 레벨의 'ai_server/weights' 폴더 안에 yolov5s.pt가 있는 경우
            # yolo_weights_path_relative = 'ai_server/weights/yolov5s.pt'
            # yolo_weights_path = script_dir / yolo_weights_path_relative
            # global_yolo_detector = yd.YOLODetector(weights_path=str(yolo_weights_path), device=str(DEVICE), img_size=640)

            # torch.hub 자동 다운로드를 사용하거나, YOLODetector 내부에서 경로 처리를 한다면 파일 이름만 전달
            global_yolo_detector = yd.YOLODetector(weights_path='yolov5s.pt', device=str(DEVICE), img_size=640) # img_size는 모델에 맞게 조정 필요

            if global_yolo_detector.model is None:
                 print("🚨 YOLO 모델 로딩이 성공하지 않았습니다. 객체 검출 기능을 사용할 수 없습니다.")
                 global_yolo_detector = None
            else:
                print("✅ YOLO 모델 로딩 완료.")

        except Exception as e:
            print(f"❌ YOLO 모델 로딩 중 예외 발생: {e}")
            global_yolo_detector = None

    print("✅ 모델 로딩 종료.")


# --- python-socketio 서버 인스턴스 생성 ---
sio = socketio.Server(cors_allowed_origins="*", ping_interval=5, ping_timeout=10, max_http_buffer_size=100000000) # 이미지 전송을 위해 버퍼 크기 증가

# WSGI 애플리케이션 생성 (SocketIO 서버를 HTTP 서버와 연결)
app = socketio.WSGIApp(sio)


# --- 이미지 처리 파이프라인 및 명령 결정 함수 ---
def process_image_and_determine_command(image_np_bgr):
    """
    OpenCV 이미지 (BGR, numpy 배열)를 입력받아,
    디헤이징 후 YOLO로 객체를 검출하고, 결과를 바탕으로 명령을 결정하는 함수

    Args:
        image_np_bgr (numpy.ndarray): OpenCV로 읽은 이미지 데이터 (BGR 형식, uint8)

    Returns:
        str or None: 라즈베리파이로 보낼 명령 문자열 ('forward', 'backward', 'stop' 등),
                     명령을 보내지 않을 경우 None 반환
    """
    print("\n--- 이미지 처리 파이프라인 시작 ---")
    command = None

    if image_np_bgr is None or image_np_bgr.size == 0:
         print("🚨 process_image: 유효하지 않은 입력 이미지입니다.")
         print("--- 이미지 처리 파이프라인 종료 (오류) ---")
         return None

    # --- 단계 1&2: 이미지 디헤이징 (모듈 함수 호출) ---
    if dehazer_module is not None and global_dehaze_net is not None:
         # dehazer_module의 apply_dehazing 함수를 호출하여 디헤이징 수행
         processed_image_np_bgr = dehazer_module.apply_dehazing(image_np_bgr, global_dehaze_net, DEVICE)
         # apply_dehazing 함수 내에서 성공/실패 메시지 출력 및 오류 처리 수행
    else:
         print("✨ 디헤이징 모듈 또는 모델이 로딩되지 않았습니다. 디헤이징 건너뜁니다.")
         processed_image_np_bgr = image_np_bgr # 디헤이징 건너뛰고 원본 이미지 사용


    # --- 단계 3&4: YOLO 객체 검출 및 결과 분석 ---
    detections = []
    annotated_img = None
    if global_yolo_detector is not None and processed_image_np_bgr is not None:
        print("🔍 YOLO 객체 검출 처리 중...")
        try:
            # YOLO Detector 인스턴스의 메소드 호출 (yolodetect 모듈에서 임포트)
            results, annotated_img = global_yolo_detector.detect_array(processed_image_np_bgr)

            if results is not None:
                 # YOLO Detector 인스턴스의 메소드 호출 (yolodetect 모듈에서 임포트)
                 detections = global_yolo_detector.extract_detections(results)
                 print(f"✅ YOLO 객체 검출 완료. 총 {len(detections)}개 객체 검출됨.")

                 # 디버깅을 위해 검출 결과가 표시된 이미지를 파일로 저장할 수 있습니다.
                 if annotated_img is not None:
                     timestamp = int(time.time())
                     output_filename = f"yolo_output_{timestamp}.jpg"
                     cv2.imwrite(output_filename, annotated_img)
                     print(f"YOLO 결과 이미지 임시 저장됨: {output_filename}")

            else:
                print("🚨 YOLO 객체 검출 결과가 없습니다.")

        except Exception as e:
            print(f"❌ YOLO 객체 검출 중 오류 발생: {e}")
            detections = []
    else:
         print("🔍 YOLO 모델이 로딩되지 않았거나 유효한 이미지가 없어 객체 검출 건너뜁니다.")


    # --- 단계 5: 검출 결과를 바탕으로 명령 결정 ---
    # TODO: 여기에 실제 명령 결정 로직을 구현해주세요.
    # 'detections' 리스트를 분석하여 원하는 조건에 따라 명령(예: 'forward', 'backward', 'stop' 등)을
    # 결정하고 'command' 변수에 할당합니다.
    print("🧠 검출 결과를 바탕으로 라즈베리파이 명령 결정 중...")

    # --- 예시 명령 결정 로직 ---
    # 실제 상황과 프로젝트 목적에 맞게 이 부분을 완전히 수정해주세요.
    person_detected = False
    car_detected = False

    if detections:
        print(f"- 검출된 객체 목록 ({len(detections)}개): {[det['class_name'] for det in detections]}")
        for det in detections:
            if det['confidence'] > 0.5:
                if det['class_name'] == 'person':
                    person_detected = True
                    print(f"   > 'person' 객체 검출됨 (신뢰도: {det['confidence']:.2f})")
                    command = "stop"
                    print(f"   -> 명령 결정: '{command}' (사람 발견)")
                    break
                elif det['class_name'] == 'car':
                     car_detected = True
                     print(f"   > 'car' 객체 검출됨 (신뢰도: {det['confidence']:.2f})")

    if command is None and car_detected:
         command = "forward"
         print(f"   -> 명령 결정: '{command}' (사람 없음, 차 발견)")

    if command is None:
        print("   -> 검출 결과에 따라 보낼 특정 명령이 없습니다.")


    print(f"➡️ 최종 결정 명령: {command}")
    print("--- 이미지 처리 파이프라인 종료 ---")

    return command


@sio.on('connect')
def handle_connect(sid, environ):
    """클라이언트 연결 시 호출"""
    print(f'✅ 클라이언트가 연결되었습니다. (SID: {sid})')


@sio.on('disconnect')
def handle_disconnect(sid):
    """클라이언트 연결 해제 시 호출"""
    print(f'❌ 클라이언트 연결이 끊어졌습니다. (SID: {sid})')


@sio.on('ack')
def handle_ack(sid, data):
    """클라이언트로부터 ACK 메시지 수신 시 호출"""
    print(f'👍 클라이언트 (SID: {sid})로부터 ACK 수신: {data}')


@sio.on('image_frame')
def handle_image_frame(sid, data):
    """
    라즈베리파이 클라이언트로부터 SocketIO를 통해 이미지 프레임 데이터를 수신하고 처리합니다.
    이미지 데이터는 Base64 문자열 형태로 전달될 것으로 예상합니다.
    """
    print(f"\n--- SocketIO 이미지 수신 핸들러 시작 (SID: {sid}) ---")
    print("📥 SocketIO 'image_frame' 이벤트로 이미지 데이터 수신")

    if 'image' not in data or not isinstance(data['image'], str):
        print("🚨 오류: 수신된 데이터에 'image' 필드가 없거나 문자열이 아닙니다.")
        sio.emit('error', {'message': 'Invalid image data format'}, room=sid)
        print("--- SocketIO 이미지 수신 핸들러 종료 (오류) ---")
        return

    base64_image_string = data['image']
    # print(f"💡 수신된 Base64 이미지 데이터 길이: {len(base64_image_string)}")


    try:
        image_bytes = base64.b64decode(base64_image_string)
        npimg = np.frombuffer(image_bytes, np.uint8)
        image_np_bgr = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if image_np_bgr is None:
             print("🚨 오류: 이미지 디코딩 실패 (cv2.imdecode)")
             sio.emit('error', {'message': 'Failed to decode image'}, room=sid)
             print("--- SocketIO 이미지 수신 핸들러 종료 (오류) ---")
             return

        print("✅ 이미지 수신 및 디코딩 완료.")

        # --- 이미지 처리 파이프라인 함수 호출 ---
        # 이미 디코딩된 이미지 (BGR numpy 배열)를 전달
        command_to_send = process_image_and_determine_command(image_np_bgr)
        # --------------------------

        if command_to_send:
            print(f"📤 클라이언트 (SID: {sid})에 '{command_to_send}' 명령 전송 시도")
            sio.emit('command', {'command': command_to_send}, room=sid)
            print(f"➡️ '{command_to_send}' 명령 전송 완료")
        else:
            print("➡️ 보낼 명령이 결정되지 않았습니다.")

        # 이미지 처리가 완료되었음을 클라이언트에게 알릴 수 있습니다. (선택 사항)
        # sio.emit('processing_done', {'status': 'success', 'command_sent': command_to_send}, room=sid)

        print("--- SocketIO 이미지 수신 핸들러 종료 (성공) ---")

    except Exception as e:
        print(f"🚨 심각한 오류 발생: 이미지 처리 중 예외 발생 - {e}")
        sio.emit('error', {'message': f'Internal server error: {e}'}, room=sid)
        print("--- SocketIO 이미지 수신 핸들러 종료 (오류) ---")
        return


# 서버 실행 진입점
if __name__ == '__main__':
    # --- 서버 시작 전에 모델들을 미리 로딩 ---
    load_models()

    # 모델 로딩 실패 여부 확인
    if global_dehaze_net is None and global_yolo_detector is None:
         print("❌ 경고: Dehazing 모델과 YOLO 모델 모두 로딩에 실패했습니다. 이미지 처리 기능이 제대로 동작하지 않을 수 있습니다.")
    elif global_dehaze_net is None:
         print("❌ 경고: Dehazing 모델 로딩에 실패했습니다. Dehazing 없이 YOLO만 실행됩니다.")
    elif global_yolo_detector is None:
         print("❌ 경고: YOLO 모델 로딩에 실패했습니다. 객체 검출 기능을 사용할 수 없습니다.")

    # --- eventlet WSGI 서버 실행 ---
    host = '0.0.0.0'
    port = 5000

    print(f"🚀 python-socketio 서버를 시작합니다 (SocketIO 이미지 수신 모드) - {host}:{port} 에서 대기...")
    eventlet.wsgi.server(eventlet.listen((host, port)), app)
