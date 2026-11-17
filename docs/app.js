// VisionVerse — Glam Studio (Web)
// Browser port of pinch_pop.py: same pinch logic, palette, filters and booth.
// Hand tracking: MediaPipe Hand Landmarker (tasks-vision) running locally in
// the browser via WebAssembly. Camera frames never leave the device.

import { FilesetResolver, HandLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.3/vision_bundle.mjs";

const MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.3/wasm";

// Same palette as pinch_pop.py (converted from BGR to CSS rgb)
const PALETTE = [
  "rgb(150,60,255)",  // electric violet
  "rgb(0,200,180)",   // teal
  "rgb(255,140,0)",   // tangerine
  "rgb(140,220,40)",  // lime
  "rgb(60,160,255)",  // sky blue
  "rgb(250,200,40)",  // gold
];
const FILTERS = ["normal", "red light", "nokia", "vhs", "b&w"];
const HAND_LINES = [
  [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17],
];
const PINCH_ON = 0.38, PINCH_OFF = 0.62, RELEASE_GRACE = 4, LOST_GRACE = 0.35;
const PEN_WIDTH = 12, ERASER_RADIUS = 44;

// ---------------------------------------------------------------- OneEuro --
class OneEuro {
  constructor(minCutoff = 1.6, beta = 0.015, dCutoff = 1.0) {
    this.mc = minCutoff; this.beta = beta; this.dc = dCutoff; this.reset();
  }
  reset() { this.x = null; this.dx = [0, 0]; }
  static alpha(cut, dt) { const tau = 1 / (2 * Math.PI * cut); return 1 / (1 + tau / dt); }
  call(p, dt) {
    dt = Math.max(dt, 1e-3);
    if (!this.x) { this.x = [p[0], p[1]]; return this.x; }
    let a = OneEuro.alpha(this.dc, dt);
    this.dx = [
      a * ((p[0] - this.x[0]) / dt) + (1 - a) * this.dx[0],
      a * ((p[1] - this.x[1]) / dt) + (1 - a) * this.dx[1],
    ];
    const cut = this.mc + this.beta * Math.hypot(this.dx[0], this.dx[1]);
    a = OneEuro.alpha(cut, dt);
    this.x = [a * p[0] + (1 - a) * this.x[0], a * p[1] + (1 - a) * this.x[1]];
    return this.x;
  }
}

// ------------------------------------------------------------------- DOM --
const $ = (s) => document.querySelector(s);
const view = $("#view"), vctx = view.getContext("2d");
const W = view.width, H = view.height;
const video = document.createElement("video");
video.playsInline = true; video.muted = true;
const drawC = document.createElement("canvas"); drawC.width = W; drawC.height = H;
const dctx = drawC.getContext("2d");
const strokeC = document.createElement("canvas"); strokeC.width = W; strokeC.height = H;
const sctx = strokeC.getContext("2d");
const smallC = document.createElement("canvas"); // for the nokia filter
const artC = document.createElement("canvas"); artC.width = W; artC.height = H;

// ----------------------------------------------------------------- state --
let landmarker = null, running = false, lastVideoTime = -1;
let swapHands = true, showSkeleton = false;
let colorI = 0, filterI = 0;
let tracks = [], nextTrackId = 0;
let curStroke = null, curOwner = null, eraseSession = false;
let undoStack = [];
let countdownEnd = null, booth = null, flashUntil = 0, toastUntil = 0;
let recorder = null, recChunks = [], recording = false;
let mouseDraw = null;

// ---------------------------------------------------------------- tracks --
function newTrack() {
  return { id: ++nextTrackId, filt: new OneEuro(), votes: [], raw: false, stable: false,
    prevStable: false, off: 0, visible: true, lastSeen: 0, role: "Right",
    points: [], cursor: [0, 0], wrist: [0, 0], uiHold: false };
}
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);

function updateTrack(t, det, now, dt) {
  const pts = det.points;
  t.points = pts; t.visible = true; t.lastSeen = now; t.wrist = pts[0];
  t.votes.push(det.label); if (t.votes.length > 20) t.votes.shift();
  const size = 0.5 * (dist(pts[0], pts[9]) + dist(pts[5], pts[17]) / 0.75) || 1;
  const ratio = dist(pts[4], pts[8]) / size;
  t.raw = ratio < (t.raw ? PINCH_OFF : PINCH_ON);
  if (t.raw) { t.stable = true; t.off = 0; }
  else if (++t.off >= RELEASE_GRACE) t.stable = false;
  const tip = [0.7 * pts[8][0] + 0.3 * pts[4][0], 0.7 * pts[8][1] + 0.3 * pts[4][1]];
  t.cursor = t.filt.call(tip, dt);
}

function updateTracks(dets, now, dt) {
  for (const t of tracks) t.visible = false;
  const pairs = [];
  dets.forEach((d, i) => tracks.forEach((t) => pairs.push([dist(t.wrist, d.points[0]), i, t])));
  pairs.sort((a, b) => a[0] - b[0]);
  const usedD = new Set(), usedT = new Set();
  for (const [d, i, t] of pairs) {
    if (usedD.has(i) || usedT.has(t.id) || d > 0.35 * W) continue;
    usedD.add(i); usedT.add(t.id); updateTrack(t, dets[i], now, dt);
  }
  dets.forEach((d, i) => { if (!usedD.has(i)) { const t = newTrack(); updateTrack(t, d, now, dt); tracks.push(t); } });
  tracks = tracks.filter((t) => now - t.lastSeen <= LOST_GRACE);
  const vis = tracks.filter((t) => t.visible);
  if (vis.length >= 2) {
    vis.sort((a, b) => a.wrist[0] - b.wrist[0]);
    vis[0].role = "Left"; vis[vis.length - 1].role = "Right";
  } else {
    for (const t of vis) {
      const r = t.votes.filter((v) => v === "Right").length;
      t.role = r * 2 >= t.votes.length ? "Right" : "Left";
    }
  }
  return tracks;
}

// ---------------------------------------------------------------- drawing --
function pushUndo() {
  const c = document.createElement("canvas"); c.width = W; c.height = H;
  c.getContext("2d").drawImage(drawC, 0, 0);
  undoStack.push(c); if (undoStack.length > 25) undoStack.shift();
}
// The live stroke is kept as points and re-rendered in two passes (glow tube,
// then white-hot core) exactly like pinch_pop.py's render_strokes — drawing
// segment-by-segment would let later segments paint over the core.
function renderCurrentStroke() {
  sctx.clearRect(0, 0, W, H);
  if (!curStroke || !curStroke.pts.length) return;
  const pts = curStroke.pts, color = PALETTE[colorI];
  sctx.save();
  sctx.lineCap = "round"; sctx.lineJoin = "round";
  const path = () => {
    sctx.beginPath();
    sctx.moveTo(pts[0][0], pts[0][1]);
    if (pts.length === 1) sctx.lineTo(pts[0][0] + 0.01, pts[0][1] + 0.01);
    else for (let i = 1; i < pts.length; i++) sctx.lineTo(pts[i][0], pts[i][1]);
  };
  sctx.shadowColor = color; sctx.shadowBlur = 18;
  sctx.strokeStyle = color; sctx.lineWidth = PEN_WIDTH; path(); sctx.stroke();
  sctx.shadowBlur = 0; sctx.strokeStyle = "rgba(255,255,255,0.9)";
  sctx.lineWidth = Math.max(2, PEN_WIDTH / 3); path(); sctx.stroke();
  sctx.restore();
}
function commitStroke() {
  if (!curStroke) return;
  dctx.drawImage(strokeC, 0, 0);
  sctx.clearRect(0, 0, W, H);
  curStroke = null;
}
function eraseAt(p) {
  dctx.save(); dctx.globalCompositeOperation = "destination-out";
  dctx.beginPath(); dctx.arc(p[0], p[1], ERASER_RADIUS, 0, Math.PI * 2); dctx.fill(); dctx.restore();
}

// ---------------------------------------------------------------- filters --
function drawVideoInto(ctx) {
  const name = FILTERS[filterI];
  ctx.save();
  if (name === "nokia") {
    const sw = W / 5, sh = H / 5;
    smallC.width = sw; smallC.height = sh;
    const sctx = smallC.getContext("2d", { willReadFrequently: true });
    sctx.save(); sctx.scale(-1, 1); sctx.drawImage(video, -sw, 0, sw, sh); sctx.restore();
    const img = sctx.getImageData(0, 0, sw, sh), d = img.data;
    const pal = [[15,56,15],[48,98,48],[139,172,15],[155,188,15]]; // BGR-ish greens -> rgb below
    for (let i = 0; i < d.length; i += 4) {
      const g = Math.min(3, ((0.299 * d[i] + 0.587 * d[i+1] + 0.114 * d[i+2]) / 64) | 0);
      const c = pal[g]; d[i] = c[2]; d[i+1] = c[1]; d[i+2] = c[0];
    }
    sctx.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(smallC, 0, 0, W, H);
    ctx.imageSmoothingEnabled = true;
    ctx.fillStyle = "rgba(0,0,0,0.10)";
    for (let x = 0; x < W; x += 5) ctx.fillRect(x, 0, 1, H);
    for (let y = 0; y < H; y += 5) ctx.fillRect(0, y, W, 1);
  } else {
    if (name === "b&w") ctx.filter = "grayscale(1) contrast(1.25) brightness(0.96)";
    else if (name === "red light") ctx.filter = "sepia(0.55) hue-rotate(-48deg) saturate(2.4) contrast(1.12) brightness(0.9)";
    ctx.translate(W, 0); ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0, W, H);
    ctx.filter = "none";
    if (name === "red light") {
      const g = ctx.createRadialGradient(W/2, H/2, H/3, W/2, H/2, H);
      g.addColorStop(0, "rgba(255,0,0,0)"); g.addColorStop(1, "rgba(120,0,0,0.45)");
      ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
    }
    if (name === "vhs") {
      ctx.globalAlpha = 0.16; ctx.drawImage(view, -7, 0); ctx.globalAlpha = 1;
      ctx.fillStyle = "rgba(0,0,0,0.16)";
      for (let y = 0; y < H; y += 3) ctx.fillRect(0, y, W, 1);
      if (Math.random() < 0.12) {
        const y = (Math.random() * (H - 30)) | 0;
        ctx.drawImage(view, 0, y, W, 12, (Math.random() * 80 - 40) | 0, y, W, 12);
      }
    }
  }
  ctx.restore();
}
function renderArt(ctx) { // clean frame: filter + drawing, no UI (like pinch_pop)
  ctx.clearRect(0, 0, W, H);
  drawVideoInto(ctx);
  ctx.drawImage(drawC, 0, 0);
  ctx.drawImage(strokeC, 0, 0);
}

// ------------------------------------------------------------- interaction --
function interact(now) {
  let penActive = false;
  for (const t of tracks) {
    const rising = t.stable && !t.prevStable;
    t.prevStable = t.stable;
    if (!t.stable) t.uiHold = false;
    if (rising && t.visible) {
      const btn = buttonAt(t.cursor);
      if (btn) { btn.click(); t.uiHold = true; }
    }
    if (t.uiHold) continue;
    if (t.role === "Right" && t.stable) {
      penActive = true;
      if (curStroke && curOwner !== t.id) commitStroke();
      if (!curStroke) {
        if (t.visible && t.raw && !buttonAt(t.cursor)) {
          pushUndo();
          curStroke = { pts: [t.cursor] };
          curOwner = t.id;
          renderCurrentStroke();
        }
      } else if (t.visible && t.raw && dist(curStroke.pts[curStroke.pts.length - 1], t.cursor) > 2.5) {
        curStroke.pts.push(t.cursor);
        renderCurrentStroke();
      }
    } else if (t.role === "Left" && t.stable && t.visible && t.raw) {
      if (!eraseSession) { pushUndo(); eraseSession = true; }
      eraseAt(t.cursor);
    }
  }
  if (!penActive && curOwner !== "mouse") commitStroke();
  if (!tracks.some((t) => t.role === "Left" && t.stable && t.raw)) eraseSession = false;
}
function buttonAt(cursor) {
  const rect = view.getBoundingClientRect();
  const cx = rect.left + (cursor[0] / W) * rect.width;
  const cy = rect.top + (cursor[1] / H) * rect.height;
  for (const b of document.querySelectorAll("#panel button, .rail button")) {
    const r = b.getBoundingClientRect();
    if (cx >= r.left - 4 && cx <= r.right + 4 && cy >= r.top - 4 && cy <= r.bottom + 4) return b;
  }
  return null;
}

// --------------------------------------------------------------- booth etc --
function makeStrip(shots) {
  const PW = 400, PH = 300, pad = 24, gap = 16, foot = 110;
  const c = document.createElement("canvas");
  c.width = PW + 2 * pad; c.height = pad + shots.length * PH + (shots.length - 1) * gap + foot;
  const x = c.getContext("2d");
  x.fillStyle = "rgb(45,25,70)"; x.fillRect(0, 0, c.width, c.height);
  shots.forEach((s, i) => {
    const cw = Math.min(s.width, (s.height * 4) / 3), x0 = (s.width - cw) / 2;
    x.drawImage(s, x0, 0, cw, s.height, pad, pad + i * (PH + gap), PW, PH);
  });
  x.fillStyle = "#fff"; x.textAlign = "center";
  x.font = "700 30px 'Segoe UI', sans-serif";
  x.fillText("PHOTO BOOTH", c.width / 2, c.height - foot + 42);
  x.font = "16px 'Segoe UI', sans-serif"; x.fillStyle = "rgb(200,150,255)";
  x.fillText(new Date().toLocaleString("en-IN", { dateStyle: "medium", timeStyle: "short" }), c.width / 2, c.height - foot + 72);
  return c;
}
function download(canvas, name) {
  canvas.toBlob((b) => {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(b); a.download = name; a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 4000);
  });
}
const stamp = () => new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
function takePhoto() {
  const c = document.createElement("canvas"); c.width = W; c.height = H;
  renderArt(c.getContext("2d"));
  download(c, `photo_${stamp()}.png`);
  flashUntil = performance.now() + 180;
  say("Photo saved");
}
function tickTimed(now) {
  const bc = $("#bigCount");
  if (countdownEnd !== null) {
    const left = countdownEnd - now;
    if (left <= 0) { countdownEnd = null; bc.classList.add("hidden"); takePhoto(); }
    else { bc.textContent = Math.ceil(left / 1000); bc.classList.remove("hidden"); }
  }
  if (booth) {
    const left = booth.t - now;
    if (left <= 0) {
      const c = document.createElement("canvas"); c.width = W; c.height = H;
      renderArt(c.getContext("2d"));
      booth.shots.push(c);
      flashUntil = now + 180;
      if (booth.shots.length >= 4) {
        const strip = makeStrip(booth.shots);
        $("#stripImg").src = strip.toDataURL("image/png");
        $("#stripDownload").href = strip.toDataURL("image/png");
        $("#stripDownload").download = `booth_${stamp()}.png`;
        $("#stripPreview").classList.remove("hidden");
        booth = null; bc.classList.add("hidden");
        say("Booth strip saved — download it below the preview");
      } else booth.t = now + 2500;
    } else {
      bc.innerHTML = `${Math.ceil(left / 1000)}<small>photo ${booth.shots.length + 1} of 4</small>`;
      bc.classList.remove("hidden");
    }
  }
}
function say(s) { const t = $("#toast"); t.textContent = s; t.classList.remove("hidden"); toastUntil = performance.now() + 2500; }

// ---------------------------------------------------------------- actions --
function press(action, idx) {
  if (action === "color") { colorI = idx; markSel("#palette button", idx); }
  else if (action === "filter") { filterI = idx; markSel("#filters button", idx); }
  else if (action === "undo") {
    const prev = undoStack.pop();
    if (prev) { dctx.clearRect(0, 0, W, H); dctx.drawImage(prev, 0, 0); curStroke = null; sctx.clearRect(0, 0, W, H); }
  }
  else if (action === "clear") { pushUndo(); dctx.clearRect(0, 0, W, H); curStroke = null; sctx.clearRect(0, 0, W, H); }
  else if (action === "photo") { if (countdownEnd === null && !booth) countdownEnd = performance.now() + 3000; }
  else if (action === "booth") {
    if (booth) { booth = null; $("#bigCount").classList.add("hidden"); say("Photo booth cancelled"); }
    else if (countdownEnd === null) { booth = { shots: [], t: performance.now() + 3000 }; say("Photo booth — strike a pose!"); }
  }
  else if (action === "rec") toggleRec();
  else if (action === "help") $("#help").classList.toggle("hidden");
  else if (action === "closeStrip") $("#stripPreview").classList.add("hidden");
}
function markSel(sel, idx) {
  document.querySelectorAll(sel).forEach((b, i) => b.classList.toggle("sel", i === idx));
}
function toggleRec() {
  if (recording) { recorder.stop(); return; }
  recChunks = [];
  const stream = view.captureStream(30);
  try { recorder = new MediaRecorder(stream, { mimeType: "video/webm" }); }
  catch { recorder = new MediaRecorder(stream); }
  recorder.ondataavailable = (e) => { if (e.data.size) recChunks.push(e.data); };
  recorder.onstop = () => {
    recording = false;
    $("#recChip").classList.add("hidden");
    document.querySelector('[data-action="rec"]').classList.remove("active");
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob(recChunks, { type: "video/webm" }));
    a.download = `video_${stamp()}.webm`; a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 4000);
    say("Video saved");
  };
  recorder.start();
  recording = true;
  $("#recChip").classList.remove("hidden");
  document.querySelector('[data-action="rec"]').classList.add("active");
  say("Recording — press V or REC to stop");
}

// ------------------------------------------------------------- hand drawing --
function drawHands(ctx) {
  for (const t of tracks) {
    if (!t.visible) continue;
    const pts = t.points;
    if (showSkeleton) {
      ctx.strokeStyle = "rgba(255,255,255,0.8)"; ctx.lineWidth = 1;
      for (const [a, b] of HAND_LINES) {
        ctx.beginPath(); ctx.moveTo(pts[a][0], pts[a][1]); ctx.lineTo(pts[b][0], pts[b][1]); ctx.stroke();
      }
    }
    ctx.fillStyle = "#fff";
    for (const i of [4, 8]) { ctx.beginPath(); ctx.arc(pts[i][0], pts[i][1], 4, 0, Math.PI * 2); ctx.fill(); }
    if (!t.raw) {
      ctx.strokeStyle = "rgba(255,255,255,0.8)"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(pts[4][0], pts[4][1]); ctx.lineTo(pts[8][0], pts[8][1]); ctx.stroke();
    }
    const [cx, cy] = t.cursor;
    ctx.font = "13px 'Segoe UI', sans-serif"; ctx.textAlign = "center";
    if (t.role === "Left") {
      ctx.strokeStyle = "#fff"; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(cx, cy, t.raw ? ERASER_RADIUS : 18, 0, Math.PI * 2); ctx.stroke();
      ctx.fillStyle = "#fff"; ctx.fillText("eraser", cx, cy - (t.raw ? ERASER_RADIUS : 18) - 10);
    } else {
      ctx.strokeStyle = t.raw ? "#fff" : PALETTE[colorI]; ctx.lineWidth = t.raw ? 2 : 3;
      ctx.beginPath(); ctx.arc(cx, cy, t.raw ? PEN_WIDTH / 2 + 3 : 18, 0, Math.PI * 2); ctx.stroke();
      ctx.fillStyle = "#fff"; ctx.fillText("pen", cx, cy - 30);
    }
  }
}

// -------------------------------------------------------------------- loop --
let prevT = performance.now();
function loop() {
  requestAnimationFrame(loop);
  if (!running) return;
  const now = performance.now();
  const dt = (now - prevT) / 1000; prevT = now;
  if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    let dets = [];
    try {
      const res = landmarker.detectForVideo(video, now);
      const lms = res.landmarks || res.handLandmarks || [];
      const hands = res.handedness || res.handednesses || [];
      dets = lms.map((lm, i) => {
        let label = hands[i] && hands[i][0] ? hands[i][0].categoryName : "Right";
        if (swapHands) label = label === "Left" ? "Right" : "Left"; // mirrored view
        return { label, points: lm.map((p) => [(1 - p.x) * W, p.y * H]) };
      });
    } catch { /* frame not ready yet */ }
    updateTracks(dets, now / 1000, dt);
  }
  interact(now / 1000);
  tickTimed(now);
  vctx.clearRect(0, 0, W, H);
  drawVideoInto(vctx);
  vctx.drawImage(drawC, 0, 0);
  vctx.drawImage(strokeC, 0, 0);
  drawHands(vctx);
  const vis = tracks.some((t) => t.visible);
  $("#statusDot").parentElement.classList.toggle("on", vis);
  $("#statusText").textContent = vis
    ? "hand detected — " + [...new Set(tracks.filter((t) => t.visible).map((t) => (t.role === "Right" ? "pen" : "eraser")))].join(", ")
    : "show your hand";
  const fl = $("#flash");
  fl.style.transition = "none"; fl.style.opacity = now < flashUntil ? 0.85 : 0;
  if (now >= toastUntil) $("#toast").classList.add("hidden");
}

// -------------------------------------------------------------------- init --
async function init() {
  const btn = $("#startBtn"), msg = $("#startMsg");
  btn.disabled = true; msg.textContent = "Loading hand-tracking model (one-time download)…";
  try {
    const fileset = await FilesetResolver.forVisionTasks(WASM_URL);
    const opts = (delegate) => ({
      baseOptions: { modelAssetPath: MODEL_URL, delegate },
      runningMode: "VIDEO", numHands: 2,
      minHandDetectionConfidence: 0.5, minHandPresenceConfidence: 0.5, minTrackingConfidence: 0.5,
    });
    try { landmarker = await HandLandmarker.createFromOptions(fileset, opts("GPU")); }
    catch { landmarker = await HandLandmarker.createFromOptions(fileset, opts("CPU")); }
  } catch (e) {
    msg.textContent = "Could not load the hand-tracking model. Check your internet connection and try again.";
    btn.disabled = false; return;
  }
  msg.textContent = "Model ready — asking for camera…";
  try {
    video.srcObject = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false,
    });
    await video.play();
  } catch (e) {
    msg.textContent = "Camera not available. Allow camera access (and close other apps using it), then try again.";
    btn.disabled = false; return;
  }
  $("#startScreen").style.display = "none";
  $("#statusText").textContent = "show your hand";
  running = true;
}

// ------------------------------------------------------------------ wiring --
PALETTE.forEach((c, i) => {
  const b = document.createElement("button");
  b.style.background = c; b.dataset.action = "color"; b.dataset.idx = i;
  b.title = "Pen colour " + (i + 1);
  if (i === 0) b.classList.add("sel");
  $("#palette").appendChild(b);
});
FILTERS.forEach((f, i) => {
  const b = document.createElement("button");
  b.textContent = f; b.dataset.action = "filter"; b.dataset.idx = i;
  if (i === 0) b.classList.add("sel");
  $("#filters").appendChild(b);
});
document.addEventListener("click", (e) => {
  const b = e.target.closest("button[data-action]");
  if (b) press(b.dataset.action, b.dataset.idx ? +b.dataset.idx : undefined);
});
$("#startBtn").addEventListener("click", init);
if (new URLSearchParams(location.search).has("autostart")) init();

view.addEventListener("pointerdown", (e) => {
  const r = view.getBoundingClientRect();
  const p = [(e.clientX - r.left) * W / r.width, (e.clientY - r.top) * H / r.height];
  pushUndo();
  mouseDraw = { erase: e.shiftKey };
  if (mouseDraw.erase) eraseAt(p);
  else { curStroke = { pts: [p] }; curOwner = "mouse"; renderCurrentStroke(); }
  view.setPointerCapture(e.pointerId);
});
view.addEventListener("pointermove", (e) => {
  if (!mouseDraw) return;
  const r = view.getBoundingClientRect();
  const p = [(e.clientX - r.left) * W / r.width, (e.clientY - r.top) * H / r.height];
  if (mouseDraw.erase) eraseAt(p);
  else if (curStroke && dist(curStroke.pts[curStroke.pts.length - 1], p) > 2.5) {
    curStroke.pts.push(p); renderCurrentStroke();
  }
});
view.addEventListener("pointerup", () => { mouseDraw = null; commitStroke(); });

document.addEventListener("keydown", (e) => {
  const k = e.key.toLowerCase();
  if (k === "z") press("undo");
  else if (k === "c") press("clear");
  else if (k === "p") press("photo");
  else if (k === "b") press("booth");
  else if (k === "v") press("rec");
  else if (k === "h") press("help");
  else if (k === "k") showSkeleton = !showSkeleton;
  else if (k === "s") { swapHands = !swapHands; tracks.forEach((t) => (t.votes = [])); say("Hands swapped"); }
  else if (k >= "1" && k <= "5") press("filter", +k - 1);
});

requestAnimationFrame(loop);
