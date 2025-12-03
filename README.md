## 1. 프로젝트 소개

본 프로젝트는 악천후 환경(비/안개/렌즈 빗방울)에서 촬영된 영상을 실시간으로 복원하고, 객체 탐지를 수행하는 AI 서버입니다.
(https://github.com/meo-yeong/derain_dedrop_dehaze) 해당 모델을 사용하였습니다.
라즈베리파이, 웹 서버, AI 서버 간 실시간 데이터 전송은 Socket.IO기반으로 이루어지며, 전송·처리 등의 파이프라인을 제공합니다.

## 2. 주요기능

### Raspberry Pi

Picamera2로 실시간 프레임(640x480) 캡처하여 전송하며 다음 3개의 스레드로 구성합니다.

-   **CaptureThread:** 카메라 프레임을 수집
-   **EncodeThread:** TurboJPEG로 인코딩
-   **SendThread:** 인코딩된 프레임을 웹서버, AI서버로 각각 Socket.IO를 통해 전송

``` python
sio.emit("video_frame", encoded) # 웹서버로 영상 전송
sio_ai.emit("image_frame", encoded) # AI서버로 영상 전송
```

### AI Server
디헤이징 모델(AOD-Net) 및 YOLOv5 객체 탐지 모델을 로딩하여,
`process_image_and_determine_command()` 함수를 통하여 다음 작업을 수행합니다.

#### 디헤이징(안개/빗방울 제거)
``` python
processed_image_np_bgr = dehazer.apply_dehazing(image_np_bgr, global_dehaze_net, DEVICE)
```

#### 객체탐지
``` python
results, annotated_img = global_yolo_detector.detect_array(processed_image_np_bgr)
```

이후 복원 + 탐지가 완료된 프레임을 웹 서버로 전송합니다.

### Web Server
Raspberry Pi에서 받아온 프레임을 브라우저로 스트리밍
``` python
@socketio.on('video_frame')
def handle_video_frame(data):
...
```
AI server에서 처리한 프레임을 브라우저로 스트리밍
``` python
@socketio.on('processed_result')
def handle_processed_result(data):
...
```

## 3. 프로젝트 구조

반드시 따라할 필요는 없음 참고만 할 것

```bash
    Autonomous-Driving/
    ├── README.md              	# 프로젝트 소개 및 사용법
    ├── docker-compose.yml     	# 전체 서비스 도커 구성 (AI 서버 + 웹서버)
    ├── .env                   	# 공통 환경변수
    ├── web_server/            	# 웹 UI + 사용자 제어 서버
    │   ├── app/
    │   │   ├── main.py        	# Flask/FastAPI 진입점
    │   │   ├── templates/     	# 웹 템플릿 (Jinja2/HTML)
    │   │   └── static/        	# JS / CSS / 이미지
    │   ├── Dockerfile         	# 웹 서버용 Docker 이미지 빌드 파일
    │   └── requirements.txt
    ├── rc_car/                	# 라즈베리파이 RC카 클라이언트
    │   ├── rasp/
    │   │   └── monitor.py     	# 카메라 프레임 전송, 센서/모터 명령 처리
    └── ai_server/             	# AI 추론 서버 (딥러닝 모델)
        ├── models/            	# 딥러닝 모델 정의(AOD-Net, YOLO 등)
        ├── checkpoints/       	# 학습된 모델(.pt)
        ├── Dockerfile         	# AI 서버 도커 이미지
        ├── aiserver.py        	# 메인 추론 서버 로직
        ├── yolodetect.py      	# YOLO 기반 객체 탐지
        └── requirements.txt
```

### 시스템 흐름
#### Raspberry Pi
-   카메라로 실시간 영상 수집
-   수집된 프레임을 AI서버로 전송

#### AI Server
-   Raspberry Pi서버에서 영상 프레임 수신
-   이미지 복원 모델 적용
    (https://github.com/meo-yeong/derain_dedrop_dehaze)
-   객체 탐지 수행
-   처리된 영상 웹 서버로 전달

#### Web Server
-   AI서버의 처리된 결과를 클라이언트(웹 브라우저, 사용자)에게 송신

### 시스템 아키텍처
<img width="725" height="420" alt="architecture" src="https://github.com/user-attachments/assets/77ccb65e-3446-4cf2-9de7-07c4249056d5" />


## 4. 필요 환경
다음 패키지는 개발 환경에서 직접 실행할때 필요로 합니다.
-   Raspberry Pi 4
-   cpu : 라이젠 5600x
-   gpu : gtx 1070
-   Python 3.11
-   Docker Desktop 4.41.2

## 5. 실행 방법
### 1) 프로젝트 루트로 이동
    cd Autonomous-Driving-main

### 2)Web Server, AI Server의 Dockerfile을 이용해서 이미지 빌드
    docker-compose build --no-cache

### 3) Web server+ AI Server 컨테이너 실행
    docker-compose up

## 6. 실행 결과
### 웹 브라우저 사용자 화면
<img width="824" height="485" alt="result" src="https://github.com/user-attachments/assets/522b1c71-b061-46e6-8606-f2e8f9ad47cb" />


## 7.개발자
<table>
  <tr>
    <td align="center">
      <a href="https://github.com/d0gn">
        <img src="https://github.com/d0gn.png" width="80" style="border-radius:50%"><br/>
        <sub><b>d0gn</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/meo-yeong">
        <img src="https://github.com/meo-yeong.png" width="80" style="border-radius:50%"><br/>
        <sub><b>meo-yeong</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/mizzcereal">
        <img src="https://github.com/mizzcereal.png" width="80" style="border-radius:50%"><br/>
        <sub><b>mizzcereal</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/jkjnaver">
        <img src="https://github.com/jkjnaver.png" width="80" style="border-radius:50%"><br/>
        <sub><b>jkjnaver</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/sick2024471">
        <img src="https://github.com/sick2024471.png" width="80" style="border-radius:50%"><br/>
        <sub><b>sick2024471</b></sub>
      </a>
    </td>
  </tr>
</table>
