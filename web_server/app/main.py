from flask import Flask, render_template, Response, request, jsonify
import threading
import time

app = Flask(__name__)

# 최신 프레임 저장 변수 (카메라 1, 2 용)
last_frame_1 = None
last_frame_2 = None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload_frame1", methods=["POST"])
def upload_frame1():
    global last_frame_1
    file = request.files.get('frame')
    print(request.files)
    if file:
        last_frame_1 = file.read()
        return 'Frame1 received', 200
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
