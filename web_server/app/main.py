# import eventlet
# eventlet.monkey_patch()

from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import base64
import torch
import threading
import time
from ai_server.models.ImageDehazing import dehaze_net
import numpy as np
import cv2

# 모델 로드 및 초기화
dehaze_net_model = dehaze_net().to("cuda")
state_dict = torch.load(
    "C:\\Users\\zmffk\\OneDrive\\바탕 화면\\AutoDrivingbranch\\ai_server\\checkpoints\\dehazer.pth",
    map_location=torch.device("cuda")
)
dehaze_net_model.load_state_dict(state_dict)
dehaze_net_model.eval()

# 모델 warm-up (성능 향상)
with torch.no_grad():
    dummy = torch.randn(1, 3, 480, 640).to("cuda")
    _ = dehaze_net_model(dummy)

# Flask 및 SocketIO 초기화
app = Flask(__name__)
socketio = SocketIO(app, async_mode='eventlet')

# 연결 클라이언트 관리
connected_clients = set()
clients_lock = threading.Lock()

# 프레임 시간 조절 (FPS 제한)
last_frame_time = 0
frame_lock = threading.Lock()

# 라우팅
@app.route("/")
def index():
    return render_template("index.html")

# 연결 이벤트
@socketio.on('connect')
def on_connect():
    with clients_lock:
        connected_clients.add(request.sid)
    print('클라이언트 연결됨:', request.sid)

@socketio.on('disconnect')
def on_disconnect():
    with clients_lock:
        connected_clients.discard(request.sid)
    print('클라이언트 연결 종료:', request.sid)

# 프레임 수신 처리
@socketio.on('video_frame')
def handle_video_frame(data):
    global last_frame_time

    with frame_lock:
        now = time.time()
        if now - last_frame_time < 0.1:  # 최대 10FPS 처리
            return
        last_frame_time = now

    try:
        frame_bytes = base64.b64decode(data)

        # 클라이언트에 원본 전송
        original_b64 = base64.b64encode(frame_bytes).decode('utf-8')
        emit('video_original', original_b64, broadcast=True)

        # # 디헤이징 처리 및 전송
        # processed_bytes = run_dehazenet_pipeline(frame_bytes)
        # processed_b64 = base64.b64encode(processed_bytes).decode('utf-8')
        # emit('video_dehazed', processed_b64, broadcast=True)

    except Exception as e:
        print(f"[ERROR] 프레임 처리 오류: {e}")

# 모드 변경 처리 (현재 미구현)
@socketio.on("change_mode")
def handle_mode_change(data):
    mode = data.get("mode")
    print(f"모드 변경 요청: {mode}")

# 수동 조작 처리
@socketio.on("manual_control")
def handle_manual_control(data):
    command = data.get("command")
    print(f"수동 제어 명령 수신: {command}")

    with clients_lock:
        for sid in connected_clients:
            socketio.emit("command", {"command": command}, to=sid)

# 전처리
def preprocess_for_model(img_np):
    img = cv2.resize(img_np, (640, 480))
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to("cuda")
    return img

# 후처리
def postprocess_model_output(output_tensor):
    output_tensor = output_tensor.squeeze(0).permute(1, 2, 0).clamp(0, 1)
    output_np = (output_tensor.cpu().numpy() * 255).astype(np.uint8)
    return output_np

# 디헤이징 파이프라인
def run_dehazenet_pipeline(frame_bytes):
    try:
        img_np = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img_np is None:
            raise ValueError("이미지 디코딩 실패")

        img_tensor = preprocess_for_model(img_np)
        with torch.no_grad():
            output_tensor = dehaze_net_model(img_tensor)
        output_np = postprocess_model_output(output_tensor)
        _, processed_jpeg = cv2.imencode('.jpg', output_np)
        return processed_jpeg.tobytes()

    except Exception as e:
        print(f"[ERROR] 디헤이징 실패: {e}")
        return frame_bytes  # 실패 시 원본 반환

# 실행
if __name__ == "__main__":
    socketio.run(app, debug=True, host="0.0.0.0", port=5000)
