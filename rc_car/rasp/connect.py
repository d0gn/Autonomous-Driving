import cv2, base64
import socketio
import time

sio = socketio.Client()
@sio.event
def connect():
    sio.emit('start_stream')

def stream_loop():
    cap = cv2.VideoCapture('sample.mp4')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        _, buffer = cv2.imencode('.jpg', frame)
        b64 = base64.b64encode(buffer).decode('utf-8')
        sio.emit('video_frame', b64)

        sio.sleep(0.03)

    cap.release()

# 메인 실행
if __name__ == '__main__':
    sio.connect('http://127.0.0.1:5000')  # 또는 main_server의 IP
    stream_loop()
