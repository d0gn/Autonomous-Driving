from flask import Flask, render_template, Response, request, jsonify

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed1")
def video_feed1():
    def generate():
        while True:
            frame = get_camera_frame_1()
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/video_feed2")
def video_feed2():
    def generate():
        while True:
            frame = get_camera_frame_2()
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
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

# 카메라 프레임 예시 (직접 구현 필요)
def get_camera_frame_1():
    return b''  # OpenCV 사용 시 프레임을 JPEG 바이트로 변환하여 반환

def get_camera_frame_2():
    return b''

if __name__ == "__main__":
    app.run(debug=True)
