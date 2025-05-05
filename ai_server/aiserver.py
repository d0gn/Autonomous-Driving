# --- 필요한 라이브러리 임포트 ---
# 표준 라이브러리
import os
import sys
import time
# import argparse # 서버 실행 시 필수는 아니지만, 필요시 사용
from pathlib import Path
#aaa
# 서드파티 라이브러리
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
from PIL import Image 
# import glob
# import base64 

# PyTorch 및 관련 라이브러리
import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
# import torch.optim # 추론 서버이므로 최적화는 필요 없음
from torchvision import transforms
script_dir = Path(__file__).parent # 현재 스크립트의 디렉토리 경로
model_dir = script_dir / 'api' # 'model' 디렉토리의 경로
sys.path.append(str(model_dir)) # sys.path에 'model' 디렉토리 경로 추가
import net 
import yolodetect as yd
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"💡 모델 추론 장치: {DEVICE}")

global_dehaze_net = None
global_yolo_detector = None

def load_models():
    """서버 시작 시 Dehazing 및 YOLO 모델을 로딩합니다."""
    global global_dehaze_net, global_yolo_detector, DEVICE

    print("⏳ 모델 로딩 시작...")

    print("⏳ Dehazing 모델 로딩 중...")
    try:
        global_dehaze_net = net.dehaze_net()
        checkpoint_path = './ai_server/checkpoints/dehazer.pth'
        if not os.path.exists(checkpoint_path):
             print(f"🚨 경고: Dehazing 체크포인트 파일이 없습니다: {checkpoint_path}")
             print("Dehazing 모델 로딩을 건너뜁니다. Dehazing 없이 YOLO만 실행됩니다.")
             global_dehaze_net = None 
        else:
            
            global_dehaze_net.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
            global_dehaze_net.to(DEVICE)
            global_dehaze_net.eval() 
            print("✅ Dehazing 모델 로딩 완료.")
    except Exception as e:
         print(f"❌ Dehazing 모델 로딩 실패: {e}")
         global_dehaze_net = None 

    
    print("⏳ YOLO 모델 로딩 중...")
    try:
        
        global_yolo_detector = yd.YOLODetector(weights_path='yolov5s.pt', device=str(DEVICE), img_size=640) 

        if global_yolo_detector.model is None:
             print("🚨 YOLO 모델 로딩이 성공하지 않았습니다. 객체 검출 기능을 사용할 수 없습니다.")
             global_yolo_detector = None # 모델 로딩 실패 시 None으로 설정
        else:
            print("✅ YOLO 모델 로딩 완료.")

    except Exception as e:
        print(f"❌ YOLO 모델 로딩 중 예외 발생: {e}")
        global_yolo_detector = None # 로딩 실패 시 None으로 설정

    print("✅ 모델 로딩 종료.")

app = Flask(__name__)

app.config['SECRET_KEY'] = 'your_safe_and_complex_secret_key_here'
# cors_allowed_origins="*": 모든 도메인의 클라이언트 접속 허용 (개발/테스트 목적)
# 실제 운영 환경에서는 특정 클라이언트 IP 또는 도메인으로 제한하는 것이 보안상 좋습니다.
socketio = SocketIO(app, cors_allowed_origins="*")


def process_image_and_determine_command(image_np_bgr):
    print("\n--- 이미지 처리 파이프라인 시작 ---")
    command = None # 기본 명령은 None (명령 없음)

    if image_np_bgr is None or image_np_bgr.size == 0:
         print("🚨 process_image: 유효하지 않은 입력 이미지입니다.")
         print("--- 이미지 처리 파이프라인 종료 (오류) ---")
         return None

    processed_image_np_bgr = image_np_bgr # 디헤이징 실패 시 원본 이미지 사용

    if global_dehaze_net is not None:
        print("✨ 이미지 디헤이징 처리 중...")
        try:
            
            image_np_rgb = cv2.cvtColor(image_np_bgr, cv2.COLOR_BGR2RGB)
            image_tensor = torch.from_numpy(image_np_rgb.copy()).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            image_tensor = image_tensor.to(DEVICE)

            with torch.no_grad(): 
                
                dehazed_tensor = global_dehaze_net(image_tensor)

            dehazed_np_rgb = dehazed_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            
            dehazed_np_rgb = (dehazed_np_rgb * 255.0).astype(np.uint8)
            
            dehazed_np_bgr = cv2.cvtColor(dehazed_np_rgb, cv2.COLOR_RGB2BGR)

            processed_image_np_bgr = dehazed_np_bgr 

            print("✅ 디헤이징 완료.")
            
            cv2.imwrite("dehazed_output.jpg", processed_image_np_bgr)
            print("디헤이징된 이미지 임시 저장됨: dehazed_output.jpg")

        except Exception as e:
            print(f"❌ 디헤이징 처리 중 오류 발생: {e}")
            
            processed_image_np_bgr = image_np_bgr
            print("디헤이징 실패, 원본 이미지로 다음 단계 진행.")
    else:
         print("✨ 디헤이징 모델이 로딩되지 않았습니다. 디헤이징 건너뜁니다.")
         processed_image_np_bgr = image_np_bgr

    # --- 단계 3&4: YOLO 객체 검출 및 결과 분석 ---
    detections = [] 
    if global_yolo_detector is not None and processed_image_np_bgr is not None:
        print("🔍 YOLO 객체 검출 처리 중...")
        try:
            
            results, annotated_img = global_yolo_detector.detect_array(processed_image_np_bgr)

            if results is not None:
                 detections = global_yolo_detector.extract_detections(results)
                 print(f"✅ YOLO 객체 검출 완료. 총 {len(detections)}개 객체 검출됨.")

                 # 디버깅을 위해 검출 결과가 표시된 이미지를 파일로 저장할 수 있습니다.
                 # if annotated_img is not None:
                 cv2.imwrite("yolo_output.jpg", annotated_img)
                 print("YOLO 결과 이미지 임시 저장됨: yolo_output.jpg")

            else:
                print("🚨 YOLO 객체 검출 결과가 없습니다.")


        except Exception as e:
            print(f"❌ YOLO 객체 검출 중 오류 발생: {e}")
            detections = [] # 오류 발생 시 검출 결과 초기화
    else:
         print("🔍 YOLO 모델이 로딩되지 않았거나 유효한 이미지가 없어 객체 검출 건너뜁니다.")



    

    print("🧠 검출 결과를 바탕으로 라즈베리파이 명령 결정 중...")

    if detections:
        print(f"- 검출된 객체 목록: {[det['class_name'] for det in detections]}")
        # 예시: 'person' 객체가 하나라도 검출되면 'stop' 명령 전송
        for det in detections:
            if det['class_name'] == 'person' and det['confidence'] > 0.5: # 신뢰도 50% 이상인 'person'
                command = "stop"
                print(f"✅ 조건 만족: '{det['class_name']}' 객체 검출 (신뢰도: {det['confidence']:.2f}). 명령: '{command}'")
                break # 'person' 찾으면 더 이상 검사 불필요 (또는 다른 객체도 고려)

        # 예시: 'car' 객체가 검출되었지만 'person'이 검출되지 않았으면 'forward' 명령 전송
        if command is None: # 아직 명령이 결정되지 않았다면
            car_found = any(det['class_name'] == 'car' and det['confidence'] > 0.5 for det in detections)
            if car_found:
                command = "forward"
                print(f"✅ 조건 만족: 'car' 객체 검출. 명령: '{command}'")


    # 모든 검사를 마쳤음에도 명령이 결정되지 않았다면 None 유지
    if command is None:
        print("✅ 조건 불만족 또는 객체 미검출. 보낼 명령이 없습니다.")


    print(f"➡️ 최종 결정 명령: {command}")
    print("--- 이미지 처리 파이프라인 종료 ---")

    return command


# ------------------------------------------------------

@socketio.on('connect')
def handle_connect():
    """클라이언트 연결 시 호출"""
    print('✅ 클라이언트가 연결되었습니다.')


@socketio.on('disconnect')
def handle_disconnect():
    """클라이언트 연결 해제 시 호출"""
    print('❌ 클라이언트 연결이 끊어졌습니다.')


@socketio.on('ack')
def handle_ack(data):
    """클라이언트로부터 ACK 메시지 수신 시 호출"""
    print(f'👍 클라이언트로부터 ACK 수신: {data}')

# HTTP POST 요청을 처리하는 라우트
@app.route('/upload_frame1', methods=['POST'])
def upload_frame():
    """라즈베리파이로부터 이미지 프레임을 수신하고 처리"""
    print("\n--- 이미지 수신 라우트 시작 ---")
    print("📥 이미지 프레임 수신 요청 받음")

   
    if 'frame' not in request.files:
        print("🚨 오류: 'frame' 파일 파트가 요청에 없습니다.")
        print("--- 이미지 수신 라우트 종료 (오류) ---")
        return jsonify({'error': 'No frame file part'}), 400

    file = request.files['frame']

    
    if file.filename == '':
        print("🚨 오류: 선택된 파일 이름이 없습니다.")
        print("--- 이미지 수신 라우트 종료 (오류) ---")
        return jsonify({'error': 'No selected file'}), 400

    if file:
        try:
            # 받은 파일 데이터를 읽고 numpy 배열로 변환 (uint8 타입)
            filestr = file.read()
            npimg = np.frombuffer(filestr, np.uint8)

            # numpy 배열을 OpenCV 이미지 형식 (BGR)으로 디코딩
            # IMREAD_COLOR는 이미지를 3채널 컬러로 읽습니다.
            image_np_bgr = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

            if image_np_bgr is None:
                 print("🚨 오류: 이미지 디코딩 실패")
                 print("--- 이미지 수신 라우트 종료 (오류) ---")
                 return jsonify({'error': 'Failed to decode image'}), 400

            print("✅ 이미지 수신 및 디코딩 완료.")

            # --- 수정: 이미지 처리 파이프라인 함수 호출 ---
            # 디코딩된 OpenCV BGR numpy 이미지를 처리 함수에 전달
            command_to_send = process_image_and_determine_command(image_np_bgr)
           

            # 이미지 처리 결과 (command_to_send 변수)에 따라 명령 전송
            if command_to_send:
                print(f"📤 클라이언트에 '{command_to_send}' 명령 전송 시도")
                # 'command' 이벤트와 함께 명령 데이터를 SocketIO로 연결된 모든 클라이언트에게 전송
                # 특정 클라이언트 (예: 이미지를 보낸 클라이언트)에게만 보내려면 request.sid 등을 활용
                socketio.emit('command', {'command': command_to_send})
                print(f"➡️ '{command_to_send}' 명령 전송 완료")
            else:
                print("➡️ 보낼 명령이 결정되지 않았습니다.")

            # 클라이언트에 HTTP 응답 반환
            print("--- 이미지 수신 라우트 종료 (성공) ---")
            return jsonify({'status': 'success', 'command_sent': command_to_send}), 200

        except Exception as e:
            # 이미지 처리 또는 전송 중 예상치 못한 오류 발생 시 처리
            print(f"🚨 심각한 오류 발생: 이미지 처리 또는 전송 중 예외 발생 - {e}")
            print("--- 이미지 수신 라우트 종료 (오류) ---")
            return jsonify({'error': str(e)}), 500

    # 파일이 존재하지 않는 예상치 못한 경우 (앞에서 이미 처리되지만, 혹시 모를 상황 대비)
    print("🚨 알 수 없는 오류 발생: 파일 처리 중 문제.")
    print("--- 이미지 수신 라우트 종료 (오류) ---")
    return jsonify({'error': 'Unknown error'}), 500

# 서버 실행 진입점
if __name__ == '__main__':
    load_models()

    # 모델 로딩 실패 여부 확인 (선택 사항, 실패 시 서버 시작을 중단할 수도 있음)
    if global_dehaze_net is None and global_yolo_detector is None:
         print("❌ 경고: Dehazing 모델과 YOLO 모델 모두 로딩에 실패했습니다. 이미지 처리 기능이 제대로 동작하지 않을 수 있습니다.")
         

    print("🚀 Flask SocketIO 서버를 시작합니다...")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)

