from flask import Flask, render_template, Response, request, jsonify
import torch
import threading
import time
from ai_server.models.ImageDehazing import dehaze_net
import numpy as np
import cv2
app = Flask(__name__)

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
def run_aodnet_pipeline(frame_bytes):
    img_np = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
    img_tensor = preprocess_for_model(img_np)
    with torch.no_grad():
        output_tensor = dehaze_net_model(img_tensor)
    output_np = postprocess_model_output(output_tensor)
    _, processed_jpeg = cv2.imencode('.jpg', output_np)
    return processed_jpeg.tobytes()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload_frame1", methods=["POST"])
def upload_frame1():
    global last_frame_1, last_frame_2
    file = request.files.get('frame')
    if file:
        frame_bytes = file.read()
        last_frame_1 = frame_bytes
        last_frame_2 = run_aodnet_pipeline(frame_bytes)
        return 'Frame1 received and processed', 200
    return 'No frame1', 400

@app.route('/favicon.ico')
def favicon():
    return '', 204  # 204 No Content 응답

@app.route("/upload_frame2", methods=["POST"])
def upload_frame2():
    global last_frame_2
    file = request.files.get('frame')
    if file:
        last_frame_2 = file.read()
        return 'Frame2 received', 200
    return 'No frame2', 400

@app.route("/video_feed1")
def video_feed1():
    def generate():
        while True:
            if last_frame_1:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + last_frame_1 + b"\r\n")
            time.sleep(0.05)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/video_feed2")
def video_feed2():
    def generate():
        while True:
            if last_frame_2:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + last_frame_2 + b"\r\n")
            time.sleep(0.05)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/mode", methods=["POST"])
def change_mode():
    mode = request.json.get("mode")
    print(f"모드 변경됨: {mode}")
    return jsonify(success=True)

@app.route("/control", methods=["POST"])
def control():
    command = request.json.get("command")
    print(f"수동 명령 수신: {command}")
    return jsonify(success=True)

# 카메라 프레임 함수는 사용하지 않음
def get_camera_frame_1():
    return b''

def get_camera_frame_2():
    return b''

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
