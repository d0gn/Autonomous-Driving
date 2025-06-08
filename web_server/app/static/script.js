const socket = io();

let mode = "auto";  // 기본 자동
let pressedKeys = new Set();  //현재 누르고 있는 키들의 집합 정의
let lastCommand = "";

document.addEventListener("DOMContentLoaded", () => {
  const autoModeBtn = document.getElementById("autoModeBtn");
  const manualModeBtn = document.getElementById("manualModeBtn");
  const manualUI = document.getElementById("manualUI");
  
    // 원본 영상 수신
  socket.on('video_original', function(data) {
    document.getElementById("video_original").src = "data:image/jpeg;base64," + data;
  });

  // 디헤이징 영상 수신
  socket.on('video_dehazed', function(data) {
  document.getElementById("video_dehazed").src = "data:image/jpeg;base64," + data;
  });

  
  // 자동 모드 버튼 클릭
  autoModeBtn.addEventListener("click", () => {
    mode = "auto";
    autoModeBtn.classList.add("selected");
    manualModeBtn.classList.remove("selected");

    manualUI.style.display = "none";
    pressedKeys.clear();
    lastCommand = "";
    socket.emit("manual_control", { command: "stop" });
    socket.emit("change_mode", { mode: "auto" });
  });

  // 수동 모드 버튼 클릭
  manualModeBtn.addEventListener("click", () => {
    mode = "manual";
    manualModeBtn.classList.add("selected");
    autoModeBtn.classList.remove("selected");

    manualUI.style.display = "block";
    socket.emit("change_mode", { mode: "manual" });
  });

  //키보드 키를 누를때
  document.addEventListener("keydown", (e) => {
    if (mode !== "manual") return;
    if(["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(e.key)){
      pressedKeys.add(e.key); //누른 키를 pressedKeys에 저장, 나중에 keyup할때 추적
      handleKeyPress();
      highlightKey(e.key);  // 🔵 가상 키보드 UI 하이라이트   
    }
  });
  
  //키보드 키를 땔때
  document.addEventListener("keyup", (e) => {
    if (mode !== "manual") return;

    if(pressedKeys.has(e.key)){
      pressedKeys.delete(e.key);  //해당 키를 땠을때 pressedKeys에서 제거
      handleKeyPress(); 
      unhighlightKey(e.key);  // 🔵 가상 키보드 UI 하이라이트
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

  // 서버로 명령 전송
  function sendCommand(cmd) {
    socket.emit("manual_control", { command: cmd });
  }

  // 🔵 가상 키보드 키 강조
  function highlightKey(key) {
    const keyMap = {
      "ArrowUp": "key-up",
      "ArrowDown": "key-down",
      "ArrowLeft": "key-left",
      "ArrowRight": "key-right"
    };
    const el = document.getElementById(keyMap[key]);
    if (el) el.classList.add("active");
  }

  // 🔵 가상 키보드 키 강조 해제
  function unhighlightKey(key) {
    const keyMap = {
      "ArrowUp": "key-up",
      "ArrowDown": "key-down",
      "ArrowLeft": "key-left",
      "ArrowRight": "key-right"
    };
    const el = document.getElementById(keyMap[key]);
    if (el) el.classList.remove("active");
  }
});
