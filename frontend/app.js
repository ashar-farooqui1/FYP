/**
 * app.js
 * ──────────────────────────────────────────────────────────────
 * ASL Translator — Frontend Logic
 *
 * Flow:
 *   1. Open webcam
 *   2. Capture frame every ~100ms
 *   3. Send base64 JPEG to /predict
 *   4. Display prediction + confidence + top-3
 *   5. Speak prediction via Web Speech API (or /tts fallback)
 * ──────────────────────────────────────────────────────────────
 */

const API_BASE      = "http://localhost:8000";
const FRAME_INTERVAL= 100;   // ms between frames sent to backend
const SPEAK_COOLDOWN= 2500;  // ms before same word is spoken again
const MIN_CONF      = 0.50;  // minimum confidence to accept prediction

// ── State ─────────────────────────────────────────────────────
let isRunning     = false;
let voiceEnabled  = true;
let frameTimer    = null;
let lastSpoken    = "";
let lastSpokenAt  = 0;
let sentence      = [];

// ── DOM refs ──────────────────────────────────────────────────
const video        = document.getElementById("video");
const overlay      = document.getElementById("overlay");
const ctx          = overlay.getContext("2d");
const statusDot    = document.getElementById("statusDot");
const statusText   = document.getElementById("statusText");
const bufferFill   = document.getElementById("bufferFill");
const bufferLabel  = document.getElementById("bufferLabel");
const handBadge    = document.getElementById("handBadge");
const handIcon     = document.getElementById("handIcon");
const handText     = document.getElementById("handText");
const predWord     = document.getElementById("predWord");
const confBar      = document.getElementById("confBar");
const confPct      = document.getElementById("confPct");
const top3List     = document.getElementById("top3List");
const sentenceDisplay = document.getElementById("sentenceDisplay");
const classesGrid  = document.getElementById("classesGrid");
const classCount   = document.getElementById("classCount");

// ── Buttons ───────────────────────────────────────────────────
document.getElementById("startBtn")  .addEventListener("click", toggleCamera);
document.getElementById("resetBtn")  .addEventListener("click", resetBuffer);
document.getElementById("muteBtn")   .addEventListener("click", toggleVoice);
document.getElementById("addWordBtn").addEventListener("click", addWordToSentence);
document.getElementById("speakBtn")  .addEventListener("click", speakSentence);
document.getElementById("clearSentBtn").addEventListener("click", clearSentence);

// ── Init ──────────────────────────────────────────────────────
checkHealth();
loadClasses();


// ─────────────────────────────────────────────────────────────
//  HEALTH CHECK
// ─────────────────────────────────────────────────────────────

async function checkHealth() {
  try {
    const res  = await fetch(`${API_BASE}/health`);
    const data = await res.json();
    if (data.status === "ok" && data.model === "loaded") {
      setStatus("online", "Backend ready");
    } else {
      setStatus("offline", "Model not loaded");
    }
  } catch {
    setStatus("offline", "Backend offline");
  }
}

function setStatus(state, text) {
  statusDot.className = `status-dot ${state}`;
  statusText.textContent = text;
}


// ─────────────────────────────────────────────────────────────
//  CLASSES
// ─────────────────────────────────────────────────────────────

async function loadClasses() {
  try {
    const res  = await fetch(`${API_BASE}/classes`);
    const data = await res.json();
    classCount.textContent = data.count;
    classesGrid.innerHTML  = data.classes
      .sort()
      .map(w => `<span class="class-chip">${w}</span>`)
      .join("");
  } catch { /* backend might not be up yet */ }
}


// ─────────────────────────────────────────────────────────────
//  CAMERA
// ─────────────────────────────────────────────────────────────

async function toggleCamera() {
  if (isRunning) {
    stopCamera();
  } else {
    await startCamera();
  }
}

async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: "user" },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();

    overlay.width  = video.videoWidth  || 640;
    overlay.height = video.videoHeight || 480;

    isRunning = true;
    document.getElementById("startBtn").textContent = "⏹ Stop";

    frameTimer = setInterval(captureAndSend, FRAME_INTERVAL);
    setStatus("online", "Translating…");
  } catch (err) {
    alert("Camera access denied: " + err.message);
  }
}

function stopCamera() {
  clearInterval(frameTimer);
  frameTimer = null;
  isRunning  = false;

  const stream = video.srcObject;
  if (stream) stream.getTracks().forEach(t => t.stop());
  video.srcObject = null;

  document.getElementById("startBtn").textContent = "▶ Start";
  setStatus("offline", "Camera stopped");
  resetUI();
}


// ─────────────────────────────────────────────────────────────
//  FRAME CAPTURE & SEND
// ─────────────────────────────────────────────────────────────

function captureFrame() {
  const canvas  = document.createElement("canvas");
  canvas.width  = video.videoWidth  || 640;
  canvas.height = video.videoHeight || 480;
  canvas.getContext("2d").drawImage(video, 0, 0);
  return canvas.toDataURL("image/jpeg", 0.6);   // compressed
}

async function captureAndSend() {
  if (!isRunning) return;

  const frame = captureFrame();

  try {
    const res  = await fetch(`${API_BASE}/predict`, {
      method : "POST",
      headers: { "Content-Type": "application/json" },
      body   : JSON.stringify({ frame }),
    });
    const data = await res.json();
    updateUI(data);
  } catch {
    setStatus("offline", "Backend unreachable");
  }
}


// ─────────────────────────────────────────────────────────────
//  UI UPDATES
// ─────────────────────────────────────────────────────────────

function updateUI(data) {
  // Buffer bar
  const pct = Math.round(data.buffer_fill * 100);
  bufferFill.style.width = `${pct}%`;
  bufferLabel.textContent = !data.hand_detected
    ? "Show your hand…"
    : pct < 100
      ? `Collecting frames… ${pct}%`
      : "Predicting…";

  // Hand detection badge
  if (data.hand_detected) {
    handBadge.classList.add("detected");
    handIcon.textContent = "✋";
    handText.textContent = "Hand detected";
  } else {
    handBadge.classList.remove("detected");
    handIcon.textContent = "🚫";
    handText.textContent = "No hand";
  }

  // Prediction
  if (data.prediction && data.confidence >= MIN_CONF) {
    predWord.textContent = data.prediction.toUpperCase();
    const confVal = Math.round(data.confidence * 100);
    confBar.style.width  = `${confVal}%`;
    confPct.textContent  = `${confVal}%`;

    // Auto-speak if new word with high confidence
    const now = Date.now();
    if (
      voiceEnabled &&
      data.prediction !== lastSpoken &&
      now - lastSpokenAt > SPEAK_COOLDOWN &&
      data.confidence >= 0.55
    ) {
      speakWord(data.prediction);
      lastSpoken   = data.prediction;
      lastSpokenAt = now;
    }
  } else if (!data.prediction) {
    // Keep last prediction visible, dim it
    confBar.style.width = "0%";
    confPct.textContent = "—";
  }

  // Top-3
  if (data.top3 && data.top3.length > 0) {
    top3List.innerHTML = data.top3.map(([word, conf], i) => `
      <li class="top3-item ${i === 0 ? "" : ""}">
        <span class="top3-item__word">${word}</span>
        <span class="top3-item__conf">${Math.round(conf * 100)}%</span>
      </li>
    `).join("");
  }
}

function resetUI() {
  predWord.textContent    = "—";
  confBar.style.width     = "0%";
  confPct.textContent     = "0%";
  top3List.innerHTML      = `<li class="top3-item top3-item--empty">—</li>`;
  bufferFill.style.width  = "0%";
  bufferLabel.textContent = "Collecting frames…";
  handBadge.classList.remove("detected");
}


// ─────────────────────────────────────────────────────────────
//  CONTROLS
// ─────────────────────────────────────────────────────────────

async function resetBuffer() {
  try {
    await fetch(`${API_BASE}/reset`, { method: "POST" });
  } catch { /* ignore */ }
  resetUI();
}

function toggleVoice() {
  voiceEnabled = !voiceEnabled;
  const btn = document.getElementById("muteBtn");
  btn.textContent = voiceEnabled ? "🔊 Voice On" : "🔇 Voice Off";
}


// ─────────────────────────────────────────────────────────────
//  SENTENCE BUILDER
// ─────────────────────────────────────────────────────────────

function addWordToSentence() {
  const word = predWord.textContent.trim();
  if (!word || word === "—") return;
  sentence.push(word);
  renderSentence();
}

function clearSentence() {
  sentence = [];
  renderSentence();
}

function renderSentence() {
  if (sentence.length === 0) {
    sentenceDisplay.innerHTML =
      `<span class="sentence-placeholder">Signs will appear here…</span>`;
  } else {
    sentenceDisplay.textContent = sentence.join(" ");
  }
}

function speakSentence() {
  const text = sentence.join(" ");
  if (text) speakWord(text);
}


// ─────────────────────────────────────────────────────────────
//  TEXT-TO-SPEECH
// ─────────────────────────────────────────────────────────────

function speakWord(text) {
  // Prefer browser Web Speech API (instant, no server round-trip)
  if ("speechSynthesis" in window) {
    window.speechSynthesis.cancel();
    const utt = new SpeechSynthesisUtterance(text);
    utt.lang  = "en-US";
    utt.rate  = 0.95;
    window.speechSynthesis.speak(utt);
    return;
  }

  // Fallback: server-side gTTS
  const audio = new Audio(`${API_BASE}/tts/${encodeURIComponent(text)}`);
  audio.play().catch(() => {});
}


// ═════════════════════════════════════════════════════════════
//  VOICE → SIGN MODE
// ═════════════════════════════════════════════════════════════

// MediaPipe hand skeleton connections
const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [0,9],[9,10],[10,11],[11,12],
  [0,13],[13,14],[14,15],[15,16],
  [0,17],[17,18],[18,19],[19,20],
  [5,9],[9,13],[13,17],[0,17],
];

// Per-joint colors (thumb=red, index=orange, middle=green, ring=blue, pinky=purple)
const JOINT_COLOR = [
  "#ef4444","#ef4444","#ef4444","#ef4444","#ef4444",
  "#f97316","#f97316","#f97316","#f97316",
  "#22c55e","#22c55e","#22c55e","#22c55e",
  "#3b82f6","#3b82f6","#3b82f6","#3b82f6",
  "#a855f7","#a855f7","#a855f7","#a855f7",
];

// ── State ─────────────────────────────────────────────────────
let signFrames    = null;
let signFrameIdx  = 0;
let signAnimTimer = null;
let micListening  = false;
let speechRec     = null;

// ── DOM ───────────────────────────────────────────────────────
const signCanvas    = document.getElementById("signCanvas");
const signCtx       = signCanvas.getContext("2d");
const voiceWordEl   = document.getElementById("voiceWord");
const frameCountEl  = document.getElementById("frameCount");
const frameSlider   = document.getElementById("frameSlider");
const playBtn       = document.getElementById("playBtn");
const signEmpty     = document.getElementById("signEmpty");

// ── Mode switching ────────────────────────────────────────────
document.getElementById("tabTranslate").addEventListener("click", () => switchMode("translate"));
document.getElementById("tabVoice")    .addEventListener("click", () => switchMode("voice"));

function switchMode(mode) {
  const isVoice = mode === "voice";
  document.getElementById("tabTranslate").classList.toggle("mode-tab--active", !isVoice);
  document.getElementById("tabVoice")    .classList.toggle("mode-tab--active",  isVoice);
  document.getElementById("translateMode").hidden = isVoice;
  document.getElementById("voiceMode")   .hidden = !isVoice;
}

// ── Mic ───────────────────────────────────────────────────────
document.getElementById("micBtn").addEventListener("click", toggleMic);

function toggleMic() {
  if (!speechRec) speechRec = buildRecognition();
  if (!speechRec) return;
  if (micListening) {
    speechRec.stop();
  } else {
    try { speechRec.start(); } catch (_) {}
  }
}

function buildRecognition() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) {
    alert("Speech recognition is not supported in this browser.\nTry Chrome or Edge.");
    return null;
  }
  const rec = new SR();
  rec.lang = "en-US";
  rec.interimResults = false;
  rec.continuous = false;

  rec.onstart = () => {
    micListening = true;
    document.getElementById("micBtn").classList.add("mic-btn--active");
    document.getElementById("micHint").textContent = "Listening…";
  };
  rec.onend = () => {
    micListening = false;
    document.getElementById("micBtn").classList.remove("mic-btn--active");
    document.getElementById("micHint").textContent = "Click the mic and say an ASL word";
  };
  rec.onerror = (e) => {
    micListening = false;
    document.getElementById("micBtn").classList.remove("mic-btn--active");
    document.getElementById("micHint").textContent = "Mic error — try again";
  };
  rec.onresult = (e) => {
    const spoken = e.results[0][0].transcript.trim().toLowerCase().split(" ")[0];
    voiceWordEl.textContent = spoken.toUpperCase();
    fetchAndShowSign(spoken);
  };
  return rec;
}

// ── Text input ────────────────────────────────────────────────
document.getElementById("voiceTextBtn").addEventListener("click", () => {
  const val = document.getElementById("voiceTextInput").value.trim().toLowerCase();
  if (!val) return;
  voiceWordEl.textContent = val.toUpperCase();
  fetchAndShowSign(val);
});
document.getElementById("voiceTextInput").addEventListener("keydown", (e) => {
  if (e.key === "Enter") document.getElementById("voiceTextBtn").click();
});

// ── Fetch sign from backend ───────────────────────────────────
async function fetchAndShowSign(word) {
  stopSignAnimation();
  signEmpty.textContent = "Loading…";
  signEmpty.style.display = "flex";
  signCtx.clearRect(0, 0, signCanvas.width, signCanvas.height);
  playBtn.disabled = true;
  frameSlider.disabled = true;
  frameCountEl.textContent = "— / —";

  try {
    const res = await fetch(`${API_BASE}/sign/${encodeURIComponent(word)}`);
    if (!res.ok) {
      signEmpty.textContent = `No sign found for "${word}"`;
      return;
    }
    const data = await res.json();
    signFrames = data.frames;
    signFrameIdx = 0;
    frameSlider.max = signFrames.length - 1;
    frameSlider.value = 0;
    frameSlider.disabled = false;
    playBtn.disabled = false;
    playBtn.textContent = "⏸ Pause";
    signEmpty.style.display = "none";
    document.getElementById("signCardLabel").textContent =
      `Sign Animation — ${word.charAt(0).toUpperCase() + word.slice(1)}`;
    startSignAnimation();
  } catch {
    signEmpty.textContent = "Backend unavailable";
  }
}

// ── Animation ─────────────────────────────────────────────────
playBtn.addEventListener("click", () => {
  if (signAnimTimer) {
    stopSignAnimation();
    playBtn.textContent = "▶ Play";
  } else {
    if (!signFrames) return;
    startSignAnimation();
    playBtn.textContent = "⏸ Pause";
  }
});

frameSlider.addEventListener("input", () => {
  stopSignAnimation();
  playBtn.textContent = "▶ Play";
  signFrameIdx = parseInt(frameSlider.value);
  renderSignFrame(signFrameIdx);
  frameCountEl.textContent = `${signFrameIdx + 1} / ${signFrames.length}`;
});

function startSignAnimation() {
  stopSignAnimation();
  const FPS = 10;
  function tick() {
    if (signFrameIdx >= signFrames.length) signFrameIdx = 0;
    renderSignFrame(signFrameIdx);
    frameSlider.value = signFrameIdx;
    frameCountEl.textContent = `${signFrameIdx + 1} / ${signFrames.length}`;
    signFrameIdx++;
  }
  tick();
  signAnimTimer = setInterval(tick, 1000 / FPS);
}

function stopSignAnimation() {
  if (signAnimTimer) { clearInterval(signAnimTimer); signAnimTimer = null; }
}

// ── Drawing ───────────────────────────────────────────────────
function renderSignFrame(idx) {
  const lm = signFrames[idx];
  if (!lm || lm.length < 21) return;

  const W = signCanvas.width;
  const H = signCanvas.height;
  const PAD = 40; // padding so hand doesn't touch edges

  signCtx.clearRect(0, 0, W, H);

  // Compute bounding box to auto-center the hand
  const xs = lm.map(p => p[0]);
  const ys = lm.map(p => p[1]);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const rangeX = maxX - minX || 1;
  const rangeY = maxY - minY || 1;
  const scale  = Math.min((W - PAD * 2) / rangeX, (H - PAD * 2) / rangeY);
  const offX   = (W - rangeX * scale) / 2 - minX * scale;
  const offY   = (H - rangeY * scale) / 2 - minY * scale;

  const pts = lm.map(([x, y]) => [x * scale + offX, y * scale + offY]);

  // Draw bones
  signCtx.lineCap  = "round";
  signCtx.lineJoin = "round";
  signCtx.lineWidth = 3;
  signCtx.strokeStyle = "#cbd5e1";
  for (const [a, b] of HAND_CONNECTIONS) {
    signCtx.beginPath();
    signCtx.moveTo(pts[a][0], pts[a][1]);
    signCtx.lineTo(pts[b][0], pts[b][1]);
    signCtx.stroke();
  }

  // Draw joints
  for (let i = 0; i < 21; i++) {
    const [x, y] = pts[i];
    const r = i === 0 ? 9 : 6;
    signCtx.beginPath();
    signCtx.arc(x, y, r, 0, Math.PI * 2);
    signCtx.fillStyle = "#ffffff";
    signCtx.fill();
    signCtx.strokeStyle = JOINT_COLOR[i] || "#2563eb";
    signCtx.lineWidth = 2.5;
    signCtx.stroke();
  }
}

// ── Populate voice chips (same classes, clickable) ────────────
async function loadVoiceChips() {
  try {
    const res  = await fetch(`${API_BASE}/classes`);
    const data = await res.json();
    const grid = document.getElementById("voiceChipsGrid");
    grid.innerHTML = data.classes.sort().map(w =>
      `<span class="class-chip voice-chip" data-word="${w}">${w}</span>`
    ).join("");
    grid.querySelectorAll(".voice-chip").forEach(chip => {
      chip.addEventListener("click", () => {
        const word = chip.dataset.word;
        voiceWordEl.textContent = word.toUpperCase();
        document.getElementById("voiceTextInput").value = word;
        fetchAndShowSign(word);
      });
    });
  } catch { /* backend might not be up yet */ }
}

loadVoiceChips();