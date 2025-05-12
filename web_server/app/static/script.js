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