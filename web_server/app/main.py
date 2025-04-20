import eventlet
eventlet.monkey_patch()

from flask import Flask, render_template, Response, request, jsonify
from flask_socketio import SocketIO, emit
import threading
import asyncio

app = Flask(__name__)

socketio = SocketIO(app, cors_allowed_origins="*")  # WebSocket 서버 활성화

# 최신 프레임 저장 변수 (카메라 1, 2 용)
last_frame_1 = None
last_frame_2 = None

# --- 라우트 설정 ---
@app.route("/")
def index():
    return render_template("index.html")

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route("/upload_frame1", methods=["POST"])
def upload_frame1():
    global last_frame_1
    file = request.files.get('frame')
    if file:
        last_frame_1 = file.read()
        return 'Frame1 received', 200
    return 'No frame1', 400

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
    return Response(frame_stream(lambda: last_frame_1), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/video_feed2")
def video_feed2():
    return Response(frame_stream(lambda: last_frame_2), mimetype="multipart/x-mixed-replace; boundary=frame")

def frame_stream(get_frame_func):
    def generate():
        while True:
            frame = get_frame_func()
            if frame:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                )
            eventlet.sleep(0.01)
    return generate()


@app.route("/mode", methods=["POST"])
def change_mode():
    mode = request.json.get("mode")
    print(f"모드 변경됨: {mode}")
    return jsonify(success=True)

clients = {}
clients_lock = threading.Lock()

@app.route("/control", methods=["POST"])
def control():
    command = request.json.get("command")
    print(f"수동 명령 수신: {command}")

    # clients dict에서 key로 sid를 꺼내 사용
    with clients_lock:
        for sid in clients:
            socketio.emit("command", {"command": command}, to=sid)

    return jsonify(success=True)

@socketio.on("connect")
def handle_connect():
    sid = request.sid  # 안전하게 저장
    clients[sid] = True
    print(f"라즈베리파이 연결됨: {sid}")

@socketio.on("disconnect")
def handle_disconnect():
    sid = request.sid
    if sid:
        with clients_lock:
            clients.pop(sid, None)
        print(f"라즈베리파이 연결 종료: {sid}")

@socketio.on("ack")
def handle_ack(data):
    print(f"라즈베리파이로부터 ack 수신: {data}")
# --- 메인 실행 ---

if __name__ == "__main__":
    socketio.run(app, debug=True, host="0.0.0.0", port=5000)
