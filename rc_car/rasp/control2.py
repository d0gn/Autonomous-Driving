# raspberry_ws_client.py
import socketio
import RPi.GPIO as GPIO
import time

# ───── GPIO 설정 ─────
IN1 = 17
IN2 = 27
ENA = 18
IN3 = 16
IN4 = 20
ENB = 12

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

motor_pins = [IN1, IN2, IN3, IN4, ENA, ENB]
for pin in motor_pins:
    GPIO.setup(pin, GPIO.OUT)

pwm_A = GPIO.PWM(ENA, 1000)
pwm_B = GPIO.PWM(ENB, 1000)
pwm_A.start(0)
pwm_B.start(0)

# ───── 동작 함수 ─────
def motor_forward(speed=60):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    pwm_A.ChangeDutyCycle(speed)
    pwm_B.ChangeDutyCycle(speed)
    print(f"→ 전진 (속도: {speed}%)")

def motor_backward(speed=60):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    pwm_A.ChangeDutyCycle(speed)
    pwm_B.ChangeDutyCycle(speed)
    print(f"← 후진 (속도: {speed}%)")

def motor_left(speed=60):
    # 왼쪽 바퀴 정지, 오른쪽 바퀴 전진
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    pwm_A.ChangeDutyCycle(0)
    pwm_B.ChangeDutyCycle(speed)
    print(f"↺ 좌회전 (속도: {speed}%)")

def motor_right(speed=60):
    # 왼쪽 바퀴 전진, 오른쪽 바퀴 정지
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)
    pwm_A.ChangeDutyCycle(speed)
    pwm_B.ChangeDutyCycle(0)
    print(f"↻ 우회전 (속도: {speed}%)")

def motor_stop():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)
    pwm_A.ChangeDutyCycle(0)
    pwm_B.ChangeDutyCycle(0)
    print("■ 정지")

# ───── WebSocket 클라이언트 ─────
sio = socketio.Client()

@sio.on("connect")
def on_connect():
    print("✅ 서버에 연결됨")

@sio.on("disconnect")
def on_disconnect():
    print("❌ 서버 연결 끊김")
    motor_stop()

@sio.on("command")
def on_command(data):
    command = data.get("command")
    print(f"📥 명령 수신: {command}")

    # 명령 처리
    if command == "forward":
        motor_forward(70)
    elif command == "backward":
        motor_backward(70)
    elif command == "left":
        motor_left(60)
    elif command == "right":
        motor_right(60)
    elif command == "stop":
        motor_stop()
    else:
        print("⚠️ 알 수 없는 명령:", command)

    # 서버에 ack 응답
    sio.emit("ack", {"status": "done", "received_command": command})

# ───── 메인 진입점 ─────
if __name__ == "__main__":
    try:
        sio.connect("http://192.168.35.9:5000")
        sio.wait()
    except KeyboardInterrupt:
        print("⛔️ 종료됨")
    finally:
        pwm_A.stop()
        pwm_B.stop()
        GPIO.cleanup()
        print("🧹 GPIO 정리 완료")
