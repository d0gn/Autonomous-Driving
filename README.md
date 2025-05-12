# 개발중 필독 !!

반드시 깃허브를 연동할땐 새로운 브랜치를 만들고 연결할 것 !!

로컬 저장소와 원격 저장소를 연동하고 나서도 본인이 작업하고 있는 폴더가 main 브랜치면 안됨

반드시 본인이 만든 브랜치로 checkout ( 브랜치 이동 ) 하고 작업해서 커밋, 푸쉬 할것

절대 main 브랜치에 커밋, 푸쉬하지 말것!!!

++
모든 폴더안에 .gitkeep 이라는 파일이 있는데 해당 파일은 폴더 구조만 생성함에 있어서 필요해서 있는거

근데 지우지 말고 본인 작업을 진행할 것. 프로젝트 마지막에 일괄제거 예정

그리고 작업할때 폴더 하나 만들어서 작업 하길 바람

ex ) ai_server/models/yolo << 이런식으로 폴더를 모듈처럼 분해 해주길 바람


## 전체적인 폴더 구성

반드시 따라할 필요는 없음 참고만 할 것

```bash
Autonomous-Driving/
├── README.md                  # 프로젝트 소개 및 사용법
├── requirements.txt           # 루트 의존성 파일 (선택사항)
├── docker-compose.yml         # 전체 서비스 Docker 구성
├── .env                       # 환경변수 설정 파일
├── .gitignore                 # Git 추적 제외 파일 설정

├── web_server/                # 사용자 웹 UI 및 명령 제어 서버
│   ├── app/
│   │   ├── main.py            # FastAPI 또는 Flask 진입점
│   │   ├── routes/            # API 라우팅 (명령, 상태 등)
│   │   └── static/            # 정적 파일 (HTML, JS, CSS)
│   ├── templates/             # Jinja2 또는 React 템플릿
│   ├── Dockerfile             # 웹 서버용 Dockerfile
│   └── requirements.txt       # 웹 서버 Python 의존성

├── rc_car/                    # 라즈베리파이용 RC카 제어 코드
│   ├── camera/
│   │   └── capture.py         # 실시간 프레임 캡처
│   ├── control/
│   │   └── motor_control.py   # PWM 기반 모터/서보 제어
│   ├── communication/
│   │   ├── send_frame.py      # 프레임 전송 (HTTP/WS)
│   │   └── receive_command.py # 제어 명령 수신
│   ├── main.py                # 전체 실행 로직
│   └── requirements.txt       # RC카 측 Python 의존성

├── ai_server/                 # AI 추론 서버 (GPU 활용)
│   ├── models/                # 개별 딥러닝 모델 정의
│   │   ├── aod_net.py         # 날씨 보정 (AOD-Net)
│   │   ├── segmentation.py    # 도로/차선 영역 분할
│   │   ├── object_detect.py   # 장애물 탐지 모델
│   │   └── controller.py      # 조향/속도 예측 컨트롤러
│   ├── inference/
│   │   └── inference_pipeline.py # 추론 처리 파이프라인
│   ├── api/
│   │   └── main.py            # AI 서버용 API 엔드포인트
│   ├── utils/
│   │   └── image_utils.py     # 전처리/후처리 유틸
│   ├── checkpoints/
│   │   └── *.pt               # 학습된 모델 파일 저장소
│   ├── Dockerfile             # AI 서버용 Dockerfile
│   └── requirements.txt       # AI 서버 의존성 목록

├── train/                     # 모델 학습 관련 코드
│   ├── datasets/              # 데이터셋 구성, 로딩, 전처리
│   ├── training_scripts/      # 각 모델 학습 스크립트
│   │   ├── train_segmentation.py
│   │   ├── train_controller.py
│   │   └── train_aod.py
│   └── config/                # 학습 설정 (YAML/JSON)

├── data/                      # 샘플 영상/이미지 및 주석
│   ├── sample_frames/         # 테스트용 프레임 이미지
│   └── annotations/           # 라벨링 정보

└── docs/                      # 문서 및 다이어그램
    ├── architecture.md        # 시스템 구성 설명
    ├── api_reference.md       # API 명세서
    └── diagrams/
        └── system_block_diagram.png  # 시스템 다이어그램
```

## 프로젝트 구성도

세부 기능을 생각해가며 개발할 것

![자율주행](https://github.com/user-attachments/assets/85092fe2-0c82-4cad-89a6-7b43f7b58bcf)

