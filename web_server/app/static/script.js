let mode = "auto";  // 기본 자동

document.addEventListener("DOMContentLoaded", () => {
  const modeBtn = document.getElementById("modeToggle");
  const manualTip = document.getElementById("manualTip");

  modeBtn.addEventListener("click", () => {
    mode = mode === "auto" ? "manual" : "auto";
    modeBtn.textContent = `모드: ${mode === "auto" ? "자동" : "수동"} (변경하려면 클릭)`;
    manualTip.style.display = mode === "manual" ? "block" : "none";

    fetch("/mode", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode }),
    });
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

    fetch("/control", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ command }),
    });
  });
});
