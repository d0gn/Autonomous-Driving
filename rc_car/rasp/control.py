# raspberry_ws_client.py
import socketio

sio = socketio.Client()

@sio.on("connect")
def on_connect():
    print("✅ 서버에 연결됨")

@sio.on("disconnect")
def on_disconnect():
    print("❌ 서버 연결 끊김")

@sio.on("command")
def on_command(data):
    command = data.get("command")
    print(f"📥 명령 수신: {command}")

    # 여기에 실제 GPIO 제어나 동작 처리
    if command == "forward":
        print("→ 전진")
    elif command == "backward":
        print("← 후진")

    # 서버에 ack 응답
    sio.emit("ack", {"status": "done", "received_command": command})

if __name__ == "__main__":
    sio.connect("http://192.168.35.9:5000")  # 서버 IP 주소로 바꿔야 함
    sio.wait()
