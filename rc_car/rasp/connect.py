import cv2
import requests
import time

SERVER_URL = 'http://<서버 IP>:5000/upload_frame1'

cap = cv2.VideoCapture('sample.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    _, buffer = cv2.imencode('.jpg', frame)
    files = {'frame': ('frame.jpg', buffer.tobytes(), 'image/jpeg')}

    try:
        response = requests.post(SERVER_URL, files=files, timeout=1)
        print('Sent frame. Server responded:', response.text)
    except requests.exceptions.RequestException as e:
        print('Error sending frame:', e)

    time.sleep(0.1)

cap.release()
