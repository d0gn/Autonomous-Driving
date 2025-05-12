from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import base64
import threading
import time

# Flask 및 SocketIO 초기화
app = Flask(__name__)
socketio = SocketIO(app, async_mode='threading')

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


# 실행
if __name__ == "__main__":
    socketio.run(app, debug=True, host="0.0.0.0", port=5000)
