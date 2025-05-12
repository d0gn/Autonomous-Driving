const socket = io();

let mode = "auto";  // 기본 자동
let pressedKeys = new Set();  //현재 누르고 있는 키들의 집합 정의
let lastCommand = "";

document.addEventListener("DOMContentLoaded", () => {
  const modeBtn = document.getElementById("modeToggle");
  const manualTip = document.getElementById("manualTip");
  
  // 원본 영상 수신
  socket.on('video_original', function(data) {
    document.getElementById("video_original").src = "data:image/jpeg;base64," + data;
  });

  // 디헤이징 영상 수신
  socket.on('video_dehazed', function(data) {
  document.getElementById("video_dehazed").src = "data:image/jpeg;base64," + data;
  });
  
  modeBtn.addEventListener("click", () => {
    mode = mode === "auto" ? "manual" : "auto";
    modeBtn.textContent = `모드: ${mode === "auto" ? "자동" : "수동"} (변경하려면 클릭)`;
    manualTip.style.display = mode === "manual" ? "block" : "none";

    if (mode === "auto"){ //자동모드로 변환시
      pressedKeys.clear();  //현재 누르고 있는 키 초기화
      socket.emit("manual_control", {command: "stop"});  //안전을 위하여 stop명령어 전송
    }

    socket.emit("change_mode", {mode});
  });

  //키보드 키를 누를때
  document.addEventListener("keydown", (e) => {
    if (mode !== "manual") return;
    if(["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(e.key)){
      pressedKeys.add(e.key); //누른 키를 pressedKeys에 저장, 나중에 keyup할때 추적
      handleKeyPress();   
    }
  });
  
  //키보드 키를 땔때
  document.addEventListener("keyup", (e) => {
    if (mode !== "manual") return;

    if(pressedKeys.has(e.key)){
      pressedKeys.delete(e.key);  //해당 키를 땠을때 pressedKeys에서 제거
      handleKeyPress(); 
    }  
  });

  function handleKeyPress() {
    const command = getCommandFromKeys(pressedKeys);

    if(command != lastCommand){ //같은 명령어 계속 보내는거 방지
      sendCommand(command);
      lastCommand = command;
    }
  }

  function getCommandFromKeys(keys) {
    const up = keys.has("ArrowUp");
    const down = keys.has("ArrowDown");
    const left = keys.has("ArrowLeft");
    const right = keys.has("ArrowRight");

    if(up && left) return "forward_left";
    if(up && right)  return  "forward_right";
    if(down && left)  return  "backward_left";
    if(down && right) return "backward_right";
    if(up) return  "forward";
    if(down)  return  "backward";
    if(left)  return  "left";
    if(right) return  "right";
    return  "stop";
  }

  function sendCommand(cmd){
    socket.emit("manual_control", {command: cmd});
  }
});