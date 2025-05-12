import base64
import time
import socketio
import cv2
from picamera2 import Picamera2
from threading import Thread
from queue import Queue
from turbojpeg import TurboJPEG

# 전역 설정
SERVER_URL = "http://<서버_IP>:5000"  # ← 여기에 서버의 IP 주소 입력
TARGET_FPS = 15

# TurboJPEG 인코더 초기화
jpeg = TurboJPEG()

# 큐 초기화
frame_queue = Queue(maxsize=10)
encoded_queue = Queue(maxsize=10)

# Socket.IO 클라이언트 초기화
sio = socketio.Client()

@sio.event
def connect():
    print("[✓] 서버에 연결되었습니다.")

@sio.event
def disconnect():
    print("[!] 서버와 연결이 끊어졌습니다.")


class CaptureThread(Thread):
    def __init__(self, camera, queue):
        super().__init__()
        self.camera = camera
        self.queue = queue
        self.running = True

    def run(self):
        while self.running:
            frame = self.camera.capture_array()
            if not self.queue.full():
                self.queue.put(frame)

    def stop(self):
        self.running = False


class EncodeThread(Thread):
    def __init__(self, in_queue, out_queue):
        super().__init__()
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.running = True

    def run(self):
        while self.running:
            if not self.in_queue.empty():
                frame = self.in_queue.get()

                # 프레임이 JPEG에 적합한지 확인 및 변환
                if len(frame.shape) == 2:
                    # 그레이스케일 → BGR
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif frame.shape[2] == 4:
                    # BGRA → BGR
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                try:
                    jpeg_bytes = jpeg.encode(frame)
                    b64_encoded = base64.b64encode(jpeg_bytes).decode('utf-8')
                    if not self.out_queue.full():
                        self.out_queue.put(b64_encoded)
                except Exception as e:
                    print(f"[X] JPEG 인코딩 오류: {e}")

    def stop(self):
        self.running = False


class SendThread(Thread):
    def __init__(self, out_queue, fps):
        super().__init__()
        self.out_queue = out_queue
        self.running = True
        self.interval = 1.0 / fps

    def run(self):
        while self.running:
            start = time.time()
            if not self.out_queue.empty():
                encoded = self.out_queue.get()
                sio.emit("video_frame", encoded)
            elapsed = time.time() - start
            time.sleep(max(0, self.interval - elapsed))

    def stop(self):
        self.running = False


def main():
    # 서버 연결
    try:
        sio.connect(SERVER_URL)
    except Exception as e:
        print("[X] 서버 연결 실패:", e)
        return

    # PiCamera 초기화
    picam2 = Picamera2()
    picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
    picam2.start()
    time.sleep(1)  # 카메라 워밍업

    # 쓰레드 시작
    capture_thread = CaptureThread(picam2, frame_queue)
    encode_thread = EncodeThread(frame_queue, encoded_queue)
    send_thread = SendThread(encoded_queue, TARGET_FPS)

    capture_thread.start()
    encode_thread.start()
    send_thread.start()

    print("[✓] 영상 전송 시작됨 (종료: Ctrl+C)")

    try:
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n[!] 종료 중...")

    finally:
        capture_thread.stop()
        encode_thread.stop()
        send_thread.stop()

        capture_thread.join()
        encode_thread.join()
        send_thread.join()

        sio.disconnect()
        print("[✓] 종료 완료")


if __name__ == "__main__":
    main()
