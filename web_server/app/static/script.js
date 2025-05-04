
/*
const socket = io();

let mode = "auto";  // 기본 자동

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

    socket.emit("change_mode", {mode});
  });

  document.addEventListener("keydown", (e) => {
    if (mode !== "manual") return;

    let command = "";
    switch (e.key) {
      case "ArrowUp": command = "forward"; break;
      case "ArrowDown": command = "backward"; break;
      case "ArrowLeft": command = "left"; break;
      case "ArrowRight": command = "right"; break;
      default: return;
    }

    socket.emit("manual_control", {command});
  });
}); 
*/

const socket = io();

let mode = "auto";  // 기본 자동
let pressedKeys = new Set();  //현재 누르고 있는 키들의 집합 정의

document.addEventListener("DOMContentLoaded", () => {
  const modeBtn = document.getElementById("modeToggle");
  const manualTip = document.getElementById("manualTip");
  

  socket.on('video_frame', function(frame_b64) {
    document.getElementById("video").src = "data:image/jpeg;base64," + frame_b64;
  });
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

    socket.emit("change_mode", {mode});
  });

  //키보드 키를 누를때
  document.addEventListener("keydown", (e) => {
    if (mode !== "manual") return;

    if (pressedKeys.has(e.key)) return; //이미 누르고 있는 키면 무시

    let command = "";
    switch (e.key) {
      case "ArrowUp": command = "forward"; break;
      case "ArrowDown": command = "backward"; break;
      case "ArrowLeft": command = "left"; break;
      case "ArrowRight": command = "right"; break;
      default: return;
    }

    pressedKeys.add(e.key); //누른 키를 pressedKeys에 저장, 나중에 keyup할때 추적   
    socket.emit("manual_control", {command});
  });
  
  //키보드 키를 땔때
  document.addEventListener("keyup", (e) => {
    if (mode !== "manual") return;

    if(pressedKeys.has(e.key)){
      pressedKeys.delete(e.key);  //해당 키를 땠을때 pressedKeys에서 제거
      socket.emit("manual_control", {command: "stop"}); //이후 stop명령 전송송   
    }
      
  });
});