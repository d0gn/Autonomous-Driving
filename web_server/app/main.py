from flask import Flask, render_template, Response, request, jsonify
from flask_socketio import SocketIO, emit
import base64
import torch
import threading
import time
from ai_server.models.ImageDehazing import dehaze_net
import numpy as np
import cv2
"""
dehaze_net_model = dehaze_net()
state_dict = torch.load("C:/Users/win/Autonomous-Driving/ai_server/checkpoints/dehazer.pth", map_location=torch.device('cpu'))
dehaze_net_model.load_state_dict(state_dict)
dehaze_net_model.eval()

# 최신 프레임 저장 변수 (카메라 1, 2 용)
last_frame_1 = None
last_frame_2 = None

#전처리 함수
def preprocess_for_model(img_np):
    img = cv2.resize(img_np, (640, 480))
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return img

#후처리 함수
def postprocess_model_output(output_tensor):
    output_tensor = output_tensor.squeeze(0).permute(1,2,0).clamp(0,1)
    output_np = (output_tensor.numpy()*255).astype(np.uint8)
    return output_np

#추론 파이프라인 함수
def run_dehazenet_pipeline(frame_bytes):
    img_np = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
    img_tensor = preprocess_for_model(img_np)
    with torch.no_grad():
        output_tensor = dehaze_net_model(img_tensor)
    output_np = postprocess_model_output(output_tensor)
    _, processed_jpeg = cv2.imencode('.jpg', output_np)
    return processed_jpeg.tobytes()
"""
app = Flask(__name__)
socketio = SocketIO(app)
connected_clients = []

@app.route("/")
def index():
    return render_template("index.html")

@socketio.on('connect')
def on_connect(auth):
    connected_clients.append(request.sid)
    print('클라이언트 연결')

@socketio.on('video_frame')
def on_video_frame(data):
    emit('video_frame', data, broadcast = True)
""""
#def handle_video_frame(data):
    global last_frame_1, last_frame_2

    # 원본 디코딩
    frame_bytes = base64.b64decode(data)
    last_frame_1 = frame_bytes

    # 원본 프레임 그대로 클라이언트 전송
    original_b64 = base64.b64encode(frame_bytes).decode('utf-8')
    emit('video_original', original_b64, broadcast=True)

    # 디헤이징 처리
    processed_bytes = run_dehazenet_pipeline(frame_bytes)
    last_frame_2 = processed_bytes

    # 디헤이징 결과 전송
    processed_b64 = base64.b64encode(processed_bytes).decode('utf-8')
    emit('video_dehazed', processed_b64, broadcast=True)
"""
@socketio.on("change_mode")
def handel_mode_changes(data):
    mode = data.get("mode")

@socketio.on("manual_control")
def handle_manual_control(data):
    command = data.get("command")
    print(f"수동 제어 명령어 : {command}")

if __name__ == "__main__":
    socketio.run(app, port=5000)
