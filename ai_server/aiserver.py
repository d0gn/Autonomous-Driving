

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
import net 

# --- 글로벌 변수 및 모델 로딩 설정 ---
# 사용할 디바이스 설정 (CUDA GPU 또는 CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"💡 모델 추론 장치: {DEVICE}")

# 모델 인스턴스를 저장할 전역 변수
global_dehaze_net = None
global_yolo_detector = None


# 별도 파일(예: yolo_detector.py)로 분리하여 임포트하는 것이 더 깔끔합니다.
# 여기서는 요청에 따라 서버 코드 파일 안에 포함시킵니다.
class YOLODetector:
    def __init__(self, weights_path='yolov5s.pt', conf_thres=0.25, img_size=640, device='cpu'): # img_size 기본값 640으로 변경
        self.device = device
        print(f"💡 YOLODetector 사용 장치: {self.device}")

        # 가중치 파일 존재 확인 (torch.hub 자동 다운로드 기능을 사용할 경우 생략 가능)
        if not os.path.exists(weights_path):
             print(f"🚨 경고: YOLO 가중치 파일이 로컬에 없습니다: {weights_path}")
             print("torch.hub에서 표준 모델 이름으로 다운로드 시도합니다.")
             # 로컬에 없으면 파일 이름만 사용하여 torch.hub 자동 다운로드 시도
             weights_path = os.path.basename(weights_path)
             if not weights_path.endswith('.pt'): # .pt 확장자가 없으면 표준 모델 이름으로 간주
                  # 예를 들어 'yolov5s' 같은 이름
                 pass # torch.hub가 알아서 다운로드할 것이라고 가정
             else: # .pt 확장자가 있는데 로컬에 없으면 문제가 있을 수 있음
                  print(f"🚨 오류: YOLO 가중치 파일({weights_path})이 로컬에 없으며 표준 모델 이름이 아닐 수 있습니다.")
                  # 오류 발생 또는 모델 로딩 실패로 이어질 수 있습니다.

        try:
            # torch.hub를 사용하여 모델 로딩
            # force_reload=True 로 설정하면 매번 모델을 새로 다운로드
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, force_reload=False)
            self.model.to(self.device)
            self.model.eval() # 추론 모드 설정 (매우 중요)
            self.model.conf = conf_thres # 객체 검출 신뢰도 임계값
            self.model.iou = 0.45 # Non-Maximum Suppression (NMS) IoU 임계값
            # YOLOv5 모델의 입력 이미지 사이즈 설정
            self.model.imgsz = img_size
            print(f"✅ YOLO 모델 로딩 완료: {weights_path}, img_size={self.model.imgsz}")
        except Exception as e:
             print(f"❌ YOLO 모델 로딩 실패: {e}")
             self.model = None # 로딩 실패 시 모델을 None으로 설정하여 이후 호출에서 오류 방지
             # 실제 운영 환경에서는 여기서 예외를 다시 발생시키거나 서버를 종료하는 것을 고려해야 합니다.


    def detect_array(self, img_array):
        """
        NumPy 배열 형태의 이미지를 입력받아 YOLO 객체 검출 수행

        Args:
            img_array (numpy.ndarray): OpenCV BGR 형식의 이미지 데이터

        Returns:
            tuple: (results object, annotated_img numpy array)
                   또는 (None, None) if model is not loaded or error occurs
        """
        if self.model is None:
            print("🚨 YOLO 모델이 로딩되지 않았습니다. 객체 검출 건너뜜.")
            return None, None

        if img_array is None or img_array.size == 0:
             print("🚨 detect_array: 유효하지 않은 입력 이미지입니다.")
             return None, None

        try:
            # YOLO 모델은 내부적으로 RGB를 사용하지만, detect_array 메소드는
            # OpenCV BGR 입력을 받아 자체적으로 변환을 처리합니다.
            results = self.model(img_array)
            # .render() 메소드는 검출 결과를 원본 이미지에 시각화하여 numpy 배열로 반환
            annotated_img = results.render()[0] # 배치 중 첫 번째 이미지 결과
            return results, annotated_img
        except Exception as e:
            print(f"❌ YOLO detect_array 중 오류 발생: {e}")
            return None, None

    def extract_detections(self, results):
        """
        YOLO 결과를 파싱하여 객체 정보를 리스트로 추출

        Args:
            results: YOLO 모델의 결과 객체 (results = self.model(img))

        Returns:
            list: 각 객체에 대한 딕셔너리 목록 (bbox, confidence, class_id, class_name)
                  또는 빈 목록 if results is None or no detections
        """
        detections = []
        if results is not None and hasattr(results, 'xyxy') and len(results.xyxy) > 0:
            # results.xyxy[0]는 배치 중 첫 번째 이미지의 검출 결과 [x1, y1, x2, y2, confidence, class_id]
            for *xyxy, conf, cls in results.xyxy[0]:
                detections.append({
                    'bbox': [float(x.item()) for x in xyxy], # 바운딩 박스 좌표 [x1, y1, x2, y2]
                    'confidence': float(conf.item()),       # 신뢰도
                    'class_id': int(cls.item()),            # 클래스 ID
                    'class_name': self.model.names[int(cls.item())] if self.model and hasattr(self.model, 'names') else f'class_{int(cls.item())}' # 클래스 이름
                })
        return detections

# --- 모델 로딩 함수 ---
def load_models():
    """서버 시작 시 Dehazing 및 YOLO 모델을 로딩합니다."""
    global global_dehaze_net, global_yolo_detector, DEVICE

    print("⏳ 모델 로딩 시작...")

    print("⏳ Dehazing 모델 로딩 중...")
    try:
        # net.py 파일에 정의된 dehaze_net() 모델 아키텍처 사용
        global_dehaze_net = net.dehaze_net()
        checkpoint_path = './checkpoints/dehazer.pth'
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
        # YOLODetector 클래스 인스턴스 생성. 가중치 파일 경로 및 디바이스 설정
        # YOLOv5s.pt는 torch.hub가 자동으로 다운로드할 수 있습니다.
        global_yolo_detector = YOLODetector(weights_path='yolov5s.pt', device=str(DEVICE), img_size=640) # img_size는 모델에 맞게 조정 필요
        # YOLODetector.__init__ 내부에서 이미 로딩 상태를 확인하고 메시지 출력
        if global_yolo_detector.model is None:
             print("🚨 YOLO 모델 로딩이 성공하지 않았습니다. 객체 검출 기능을 사용할 수 없습니다.")
             global_yolo_detector = None # 모델 로딩 실패 시 None으로 설정
        else:
            print("✅ YOLO 모델 로딩 완료.")

    except Exception as e:
        print(f"❌ YOLO 모델 로딩 중 예외 발생: {e}")
        global_yolo_detector = None # 로딩 실패 시 None으로 설정

    print("✅ 모델 로딩 종료.")


# Flask 애플리케이션 및 SocketIO 초기화
app = Flask(__name__)

app.config['SECRET_KEY'] = 'your_safe_and_complex_secret_key_here'
# cors_allowed_origins="*": 모든 도메인의 클라이언트 접속 허용 (개발/테스트 목적)
# 실제 운영 환경에서는 특정 클라이언트 IP 또는 도메인으로 제한하는 것이 보안상 좋습니다.
socketio = SocketIO(app, cors_allowed_origins="*")


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
    command = None # 기본 명령은 None (명령 없음)

    if image_np_bgr is None or image_np_bgr.size == 0:
         print("🚨 process_image: 유효하지 않은 입력 이미지입니다.")
         print("--- 이미지 처리 파이프라인 종료 (오류) ---")
         return None

    processed_image_np_bgr = image_np_bgr # 디헤이징 실패 시 원본 이미지 사용

    # --- 단계 1&2: 이미지 디헤이징 ---
    if global_dehaze_net is not None:
        print("✨ 이미지 디헤이징 처리 중...")
        try:
            # OpenCV BGR (HxWx3) numpy 배열을 PyTorch 입력 형식으로 변환
            # BGR -> RGB
            image_np_rgb = cv2.cvtColor(image_np_bgr, cv2.COLOR_BGR2RGB)
            # HxWx3 numpy -> CxHxW tensor, 정규화 [0, 1]
            image_tensor = torch.from_numpy(image_np_rgb.copy()).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            image_tensor = image_tensor.to(DEVICE)

            # 디헤이징 모델 적용
            with torch.no_grad(): # 추론 시에는 그래디언트 계산 비활성화
                # 모델 출력은 일반적으로 [0, 1] 범위의 RGB Tensor (NCHW)
                dehazed_tensor = global_dehaze_net(image_tensor)

            # 디헤이징 결과 Tensor를 OpenCV BGR numpy 배열로 변환
            # 배치 차원 제거, CxHxW -> HxWx3
            dehazed_np_rgb = dehazed_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            # [0, 1] 범위 -> [0, 255] 범위, uint8 타입으로 변환
            dehazed_np_rgb = (dehazed_np_rgb * 255.0).astype(np.uint8)
            # RGB -> BGR
            dehazed_np_bgr = cv2.cvtColor(dehazed_np_rgb, cv2.COLOR_RGB2BGR)

            processed_image_np_bgr = dehazed_np_bgr # 디헤이징 성공 시 결과 이미지 사용

            print("✅ 디헤이징 완료.")
            # 디버깅을 위해 디헤이징된 이미지를 파일로 저장할 수 있습니다.
            # cv2.imwrite("dehazed_output.jpg", processed_image_np_bgr)
            # print("디헤이징된 이미지 임시 저장됨: dehazed_output.jpg")

        except Exception as e:
            print(f"❌ 디헤이징 처리 중 오류 발생: {e}")
            # 오류 발생 시 원본 이미지를 그대로 사용
            processed_image_np_bgr = image_np_bgr
            print("디헤이징 실패, 원본 이미지로 다음 단계 진행.")
    else:
         print("✨ 디헤이징 모델이 로딩되지 않았습니다. 디헤이징 건너뜁니다.")
         processed_image_np_bgr = image_np_bgr # 모델 없으면 원본 사용

    # --- 단계 3&4: YOLO 객체 검출 및 결과 분석 ---
    detections = [] # 검출된 객체 정보를 저장할 리스트
    if global_yolo_detector is not None and processed_image_np_bgr is not None:
        print("🔍 YOLO 객체 검출 처리 중...")
        try:
            # 디헤이징된 (또는 원본) 이미지를 YOLO Detector에 전달
            results, annotated_img = global_yolo_detector.detect_array(processed_image_np_bgr)

            if results is not None:
                 # 검출 결과를 파싱하여 필요한 정보 추출
                 detections = global_yolo_detector.extract_detections(results)
                 print(f"✅ YOLO 객체 검출 완료. 총 {len(detections)}개 객체 검출됨.")

                 # 디버깅을 위해 검출 결과가 표시된 이미지를 파일로 저장할 수 있습니다.
                 # if annotated_img is not None:
                 #     cv2.imwrite("yolo_output.jpg", annotated_img)
                 #     print("YOLO 결과 이미지 임시 저장됨: yolo_output.jpg")

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


# SocketIO 이벤트 핸들러 (이전 코드와 동일)
@socketio.on('connect')
def handle_connect():
    """클라이언트 연결 시 호출"""
    print('✅ 클라이언트가 연결되었습니다.')
    # 연결된 클라이언트의 sid (세션 ID)를 저장해두면 특정 클라이언트에만 메시지 전송 가능
    # global connected_client_sid # 예시: 전역 변수에 저장
    # connected_client_sid = request.sid
    # print(f"연결된 클라이언트 SID: {connected_client_sid}")


@socketio.on('disconnect')
def handle_disconnect():
    """클라이언트 연결 해제 시 호출"""
    print('❌ 클라이언트 연결이 끊어졌습니다.')


@socketio.on('ack')
def handle_ack(data):
    """클라이언트로부터 ACK 메시지 수신 시 호출"""
    print(f'👍 클라이언트로부터 ACK 수신: {data}')

# HTTP POST 요청을 처리하는 라우트 (수정됨: 이미지 처리 파이프라인 함수 호출)
@app.route('/upload_frame1', methods=['POST'])
def upload_frame():
    """라즈베리파이로부터 이미지 프레임을 수신하고 처리"""
    print("\n--- 이미지 수신 라우트 시작 ---")
    print("📥 이미지 프레임 수신 요청 받음")

    # HTTP 요청에 'frame' 이름으로 파일 데이터가 있는지 확인
    if 'frame' not in request.files:
        print("🚨 오류: 'frame' 파일 파트가 요청에 없습니다.")
        print("--- 이미지 수신 라우트 종료 (오류) ---")
        return jsonify({'error': 'No frame file part'}), 400

    file = request.files['frame']

    # 파일 이름이 비어있는지 확인
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
            # --------------------------

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
         # raise SystemExit("필수 모델 로딩 실패. 서버 시작 중단.") # 필요시 주석 해제하여 서버 시작 중단

    print("🚀 Flask SocketIO 서버를 시작합니다...")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)

