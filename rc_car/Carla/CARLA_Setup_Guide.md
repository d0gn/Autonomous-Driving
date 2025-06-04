
# CARLA 시뮬레이터 설치 및 실행 가이드 (Windows + Python 3.7)

## 1. CARLA 시뮬레이터 설치

- **공식 깃허브 링크**: [https://github.com/carla-simulator/carla/releases](https://github.com/carla-simulator/carla/releases)
- **버전 선택**: `CARLA_0.9.15.zip` (Windows용)
- **설치 방법**:
  1. 위 링크에서 `CARLA_0.9.15.zip` 다운로드
  2. `C:/` 경로에 압축 해제
  3. `CarlaUE4.exe` 실행

## 2. Python 3.7 설치

- **공식 다운로드 링크**: [https://www.python.org/downloads/release/python-379/](https://www.python.org/downloads/release/python-379/)

- **설치 시 주의사항**:
  1. 설치 시 `Add Python to PATH` 반드시 체크
  2. "Customize installation" 선택
  3. 설치 경로: `C:/Program Files/Python37`

## 3. Python 가상환경 생성 및 설정

```bash
# 가상환경 생성 (Python 3.7 경로는 환경에 따라 다를 수 있음)
"C:\Program Files\Python37\python.exe" -m venv venv37

# 가상환경 실행
.\venv37\Scripts\activate

# 가상환경 탈출
deactivate

# 가상환경 삭제
rmdir /s /q venv37
```

## 4. 필수 패키지 설치 (`requirements.txt`)

`requirements.txt`에 다음과 같은 형식으로 작성되어 있어야 합니다:

```
numpy
opencv-python
<carla_whl_파일_절대경로>
```

예시:
```
C:\CARLA_0.9.15\WindowsNoEditor\PythonAPI\carla\dist\carla-0.9.15-cp37-cp37m-win_amd64.whl
```

### 설치 명령어:

```bash
pip install -r requirements.txt
```

## 5. CARLA 시뮬레이터 스크립트 실행

```bash
# 실행 파일 실행
python carlaspawner.py

# 또는
python3 carlaspawner.py
```
