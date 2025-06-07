import os
import sys
import time
# import argparse
from pathlib import Path
import base64 

import socketio 
import eventlet 
import eventlet.wsgi 

import cv2 
import numpy as np 
from PIL import Image 
# import glob

import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
from torchvision import transforms 
import models.derainhaze as dh
# import torch.optim
import api.dehazer as dehazer
import models.net as net
import yolodetect as yd

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"모델에서 사용하는 장치 {DEVICE}")
global_dehaze_net = None
global_yolo_detector = None

# --- 모델 로딩 함수 ---
def load_models():
    global global_dehaze_net, global_yolo_detector, DEVICE

    print("모델 로딩 시작")
    print("디헤이징 로딩 ")

    try:
        checkpoint_path = './epoch60+datasetplus.pt'

        print(f"디헤이징 체크포인트 경로 {checkpoint_path}")

        if not os.path.exists(checkpoint_path):
                print(f"디헤이징 체크포인트 확인 실패 {checkpoint_path}")
                global_dehaze_net = None
        else:
            global_dehaze_net = dh.DerainNet()
            global_dehaze_net = torch.jit.load(checkpoint_path, map_location=DEVICE)
            global_dehaze_net.to(DEVICE)
            global_dehaze_net.eval()
            print("디헤이징 로딩 완료")
    except Exception as e:
            print(f"디헤이징 로딩 실패: {e}")
            global_dehaze_net = None

    print("YOLO 로딩")
    if yd is None:
         print("yolo 모듈 임포트 실패")
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
                 print("YOLO 로딩 실패")
                 global_yolo_detector = None
            else:
                print("YOLO 로딩 성공 ")

        except Exception as e:
            print(f"YOLO 로딩 실패  {e}")
            global_yolo_detector = None

    print("모델 로딩 종료 ")


# --- python-socketio 서버 인스턴스 생성 ---
sio = socketio.Server(cors_allowed_origins="*", ping_interval=5, ping_timeout=10, max_http_buffer_size=100000000) # 이미지 전송을 위해 버퍼 크기 증가

# WSGI 애플리케이션 생성
app = socketio.WSGIApp(sio)

#web_server(main.py)에 연결할 클라이언트
web_sio = socketio.Client()

#web_server에 연결
web_sio.connect('http://127.0.0.1:5000')

# --- 이미지 처리 파이프라인 및 명령 결정 함수 ---
def process_image_and_determine_command(image_np_bgr):
    print("이미지 처리 ")
    command = None

    if image_np_bgr is None or image_np_bgr.size == 0:
         print("유효하지 않은 이미지")
         print("이미지 처리 종료 (오류)")
         return None

    # --- 단계 1&2: 이미지 디헤이징 (모듈 함수 호출) ---

    processed_image_np_bgr = dehazer.apply_dehazing(image_np_bgr, global_dehaze_net, DEVICE)



    # --- 단계 3&4: YOLO 객체 검출 및 결과 분석 ---
    detections = []
    annotated_img = None
    if global_yolo_detector is not None and processed_image_np_bgr is not None:
        print("YOLO 처리 ")
        try:
            # YOLO Detector 인스턴스의 메소드 호출 (yolodetect 모듈에서 임포트)
            results, annotated_img = global_yolo_detector.detect_array(processed_image_np_bgr)

            if results is not None:
                 # YOLO Detector 인스턴스의 메소드 호출 (yolodetect 모듈에서 임포트)
                 detections = global_yolo_detector.extract_detections(results)
                 print(f"YOLO 객체 검출 {len(detections)}개 검출됨.")

                 # 디버깅을 위해 검출 결과가 표시된 이미지를 파일로 저장할 수 있습니다.
                 #if annotated_img is not None:
                     #timestamp = int(time.time())
                     #output_filename = f"yolo_output_{timestamp}.jpg"
                     #cv2.imwrite(output_filename, annotated_img)
                     #print(f"YOLO 결과 저장 {output_filename}")

            else:
                print("YOLO 검출 결과 없음 ")

        except Exception as e:
            print(f"YOLO 오류 발생 {e}")
            detections = []
    else:
         print("yolo 없음 건너뜀 ")


    # --- 단계 5: 검출 결과를 바탕으로 명령 결정 ---
    # 'detections' 리스트를 분석하여 원하는 조건에 따라 명령(예: 'forward', 'backward', 'stop' 등)을
    # 결정하고 'command' 변수에 할당
    print("명령 결정 ")

    # --- 예시 명령 결정 로직 ---

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
        print("   -> 명령 x ")


    print(f"결정 명령: {command}")
    print("이미지 처리 종료료")

    return command, processed_image_np_bgr


@sio.on('connect')
def handle_connect(sid, environ):
    """클라이언트 연결 시 호출"""
    print(f'클라이언트 연결  (SID: {sid})')


@sio.on('disconnect')
def handle_disconnect(sid):
    """클라이언트 연결 해제 시 호출"""
    print(f'클라이언트 연결 헤제 (SID: {sid})')


@sio.on('ack')
def handle_ack(sid, data):
    """클라이언트로부터 ACK 메시지 수신 시 호출"""
    print(f'클라이언트 (SID: {sid})로부터 ACK 수신: {data}')



'''def handle_image_frame(sid, data):
    print(f"\n--- SocketIO 이미지 수신 핸들러 시작 (SID: {sid}) ---")
    print("📥 SocketIO 'image_frame' 이벤트로 이미지 데이터 수신")

    base64_image_string = data
    print(f"💡 수신된 Base64 이미지 데이터 길이: {len(base64_image_string)}")


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
        command_to_send, processed_image  = process_image_and_determine_command(image_np_bgr)
        # --------------------------
        
        # 처리 결과와 이미지(재인코딩된 Base64) 함께 보내기
        _, jpeg_encoded = cv2.imencode('.jpg', processed_image)
        jpeg_base64 = base64.b64encode(jpeg_encoded.tobytes()).decode('utf-8')
        web_sio.emit('processed_result', {
    'command': command_to_send,
    'frame': jpeg_base64
})
        
        if command_to_send:
            print(f"📤 클라이언트 (SID: {sid})에 '{command_to_send}' 명령 전송 시도")
            sio.emit('command', {'command': command_to_send}, room=sid)
            print(f"➡️ '{command_to_send}' 명령 전송 완료")
        else:
            print("➡️ 보낼 명령이 결정되지 않았습니다.")

        # sio.emit('processing_done', {'status': 'success', 'command_sent': command_to_send}, room=sid)

        print("--- SocketIO 이미지 수신 핸들러 종료 (성공) ---")

    except Exception as e:
        print(f"🚨 심각한 오류 발생: 이미지 처리 중 예외 발생 - {e}")
        sio.emit('error', {'message': f'Internal server error: {e}'}, room=sid)
        print("--- SocketIO 이미지 수신 핸들러 종료 (오류) ---")
        return'''
@sio.on('image_frame')
def handle_image_frame(sid, data):
    print(f"\n--- SocketIO 이미지 수신 핸들러 시작 (SID: {sid}) ---")
    print("📥 SocketIO 'image_frame' 이벤트로 이미지 데이터 수신")

    base64_image_string = data
    print(f"💡 수신된 Base64 이미지 데이터 길이: {len(base64_image_string)}")

    try:
        # Base64 → NumPy 이미지 디코딩
        image_bytes = base64.b64decode(base64_image_string)
        npimg = np.frombuffer(image_bytes, np.uint8)
        image_np_bgr = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if image_np_bgr is None:
            print("🚨 오류: 이미지 디코딩 실패 (cv2.imdecode)")
            sio.emit('error', {'message': 'Failed to decode image'}, room=sid)
            print("--- SocketIO 이미지 수신 핸들러 종료 (오류) ---")
            return

        print("이미지 수신 및 디코딩 완료.")

        # 이미지 처리 → 명령 결정 및 처리 이미지 생성
        command_to_send, processed_image = process_image_and_determine_command(image_np_bgr)

        # 원본 & 처리 이미지 모두 Base64 인코딩
        _, original_encoded = cv2.imencode('.jpg', image_np_bgr)
        original_base64 = base64.b64encode(original_encoded.tobytes()).decode('utf-8')

        _, processed_encoded = cv2.imencode('.jpg', processed_image)
        processed_base64 = base64.b64encode(processed_encoded.tobytes()).decode('utf-8')

        # 웹서버(web_sio)로 전송
        web_sio.emit('processed_result', {
            'command': command_to_send,
            'original_frame': original_base64,
            'processed_frame': processed_base64
        })

        # 클라이언트에게 명령 전송
        if command_to_send:
            print(f"📤 클라이언트 (SID: {sid})에 '{command_to_send}' 명령 전송 시도")
            sio.emit('command', {'command': command_to_send}, room=sid)
            print(f"➡️ '{command_to_send}' 명령 전송 완료")
        else:
            print("➡️ 보낼 명령이 결정되지 않았습니다.")

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
         print("YOLO 디헤이징 모두 로딩 실패 ")
    elif global_dehaze_net is None:
         print("디헤이징 실패 YOLO만 시행")
    elif global_yolo_detector is None:
         print("YOLO 실패 객체검출 불가 ")

    # --- eventlet WSGI 서버 실행 ---
    host = '0.0.0.0'
    port = 5001

    print(f"서버를 시작 - {host}:{port} 에서 대기...")
    eventlet.wsgi.server(eventlet.listen((host, port)), app)
