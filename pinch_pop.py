import math
import os
import time
import urllib.request
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "hand_landmarker.task")
CAPTURE_DIR = os.path.join(HERE, "captures")

CAMERA_INDEX = 0
FRAME_W, FRAME_H = 1280, 720
SWAP_HANDS = True            # MediaPipe labels a mirrored image the other way round
PINCH_ON, PINCH_OFF = 0.38, 0.62   # thumb-index gap / hand size (hysteresis)
RELEASE_GRACE = 4            # frames of "not pinching" before a stroke ends
LOST_GRACE = 0.35            # seconds a hand may vanish without ending its stroke
PEN_WIDTH = 12               # neon pen thickness (px)
ERASER_RADIUS = 44

# Photo booth
BOOTH_SHOTS = 4              # photos per strip
BOOTH_FIRST = 3.0            # seconds before the first shot
BOOTH_GAP = 2.5              # seconds between shots
BOOTH_PREVIEW = 6.0          # seconds the finished strip stays on screen
BOOTH_BG = (70, 25, 45)      # strip background (BGR deep purple)

ACCENT = (255, 90, 160)      # UI accent (BGR violet)
REC_RED = (40, 40, 230)      # recording dot

PALETTE = [                   # BGR neon shades
    (255, 60, 150),    # electric violet
    (180, 200, 0),     # teal
    (0, 140, 255),     # tangerine
    (40, 220, 140),    # lime
    (255, 160, 60),    # sky blue
    (40, 200, 250),    # gold
]
FILTERS = ["normal", "red light", "nokia", "vhs", "b&w"]

HAND_LINES = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]
FONT = cv2.FONT_HERSHEY_SIMPLEX


# ----------------------------------------------------------------------------
# Smoothing
# ----------------------------------------------------------------------------
class OneEuro:
    """One-Euro filter: smooth when moving slowly (no jitter), responsive when fast."""

    def __init__(self, min_cutoff=1.6, beta=0.015, d_cutoff=1.0):
        self.mc, self.beta, self.dc = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self):
        self.x = None
        self.dx = np.zeros(2)

    @staticmethod
    def _alpha(cut, dt):
        tau = 1.0 / (2 * math.pi * cut)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, p, dt):
        p = np.asarray(p, float)
        dt = max(dt, 1e-3)
        if self.x is None:
            self.x = p
            return p
        a = self._alpha(self.dc, dt)
        self.dx = a * ((p - self.x) / dt) + (1 - a) * self.dx
        cut = self.mc + self.beta * np.linalg.norm(self.dx)
        a = self._alpha(cut, dt)
        self.x = a * p + (1 - a) * self.x
        return self.x


def chaikin(pts, iters=2):
    """Corner-cutting curve smoothing so strokes look like clean curves."""
    p = np.asarray(pts, np.float32)
    for _ in range(iters):
        if len(p) < 3:
            break
        q = 0.75 * p[:-1] + 0.25 * p[1:]
        r = 0.25 * p[:-1] + 0.75 * p[1:]
        out = np.empty((2 * len(q), 2), np.float32)
        out[0::2], out[1::2] = q, r
        p = np.vstack([p[:1], out, p[-1:]])
    return p


# ----------------------------------------------------------------------------
# Hand tracking (MediaPipe) + stable tracks
# ----------------------------------------------------------------------------
def ensure_model():
    if os.path.exists(MODEL_PATH):
        return
    print("Downloading hand tracking model (one time only)...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    except Exception as e:
        raise SystemExit(
            f"Could not download the model: {e}\nDownload it from:\n  {MODEL_URL}\n"
            f"and save it as:\n  {MODEL_PATH}"
        )


class Detector:
    def __init__(self):
        ensure_model()
        opts = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.lm = vision.HandLandmarker.create_from_options(opts)
        self.t0 = time.monotonic()
        self.last_ts = -1

    def detect(self, mirrored_bgr):
        h, w = mirrored_bgr.shape[:2]
        small = cv2.resize(mirrored_bgr, (640, int(640 * h / w)), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = max(int((time.monotonic() - self.t0) * 1000), self.last_ts + 1)
        self.last_ts = ts
        res = self.lm.detect_for_video(img, ts)
        dets = []
        for lms, handed in zip(res.hand_landmarks, res.handedness):
            label = handed[0].category_name
            if SWAP_HANDS_STATE["swap"]:
                label = "Left" if label == "Right" else "Right"
            pts = [(l.x * w, l.y * h) for l in lms]  # landmarks are normalised -> full-res px
            dets.append({"label": label, "points": pts})
        return dets

    def close(self):
        self.lm.close()


SWAP_HANDS_STATE = {"swap": SWAP_HANDS}


class Track:
    """One physical hand, followed across frames (stable id, role and pinch state)."""
    _next = 0

    def __init__(self):
        Track._next += 1
        self.id = Track._next
        self.votes = deque(maxlen=20)
        self.filt = OneEuro()
        self.raw = False          # pinching right now (hysteresis)
        self.stable = False       # pinching, with release grace (no flicker)
        self.prev_stable = False
        self.off = 0
        self.visible = True
        self.last_seen = 0.0
        self.role = "Right"
        self.points = []
        self.cursor = (0, 0)
        self.wrist = (0, 0)
        self.ui_hold = False

    def update(self, det, now, dt):
        pts = det["points"]
        self.points, self.visible, self.last_seen = pts, True, now
        self.wrist = pts[0]
        self.votes.append(det["label"])
        d = math.dist
        size = 0.5 * (d(pts[0], pts[9]) + d(pts[5], pts[17]) / 0.75) or 1.0
        ratio = d(pts[4], pts[8]) / size
        self.raw = ratio < (PINCH_OFF if self.raw else PINCH_ON)
        if self.raw:
            self.stable, self.off = True, 0
        else:
            self.off += 1
            if self.off >= RELEASE_GRACE:
                self.stable = False
        tip = (0.7 * pts[8][0] + 0.3 * pts[4][0], 0.7 * pts[8][1] + 0.3 * pts[4][1])
        self.cursor = tuple(self.filt(tip, dt))


class HandManager:
    def __init__(self):
        self.tracks = []

    def update(self, dets, now, dt, w):
        for t in self.tracks:
            t.visible = False
        pairs = sorted(
            ((math.dist(t.wrist, d["points"][0]), i, t) for i, d in enumerate(dets) for t in self.tracks),
            key=lambda x: x[0],
        )
        used_d, used_t = set(), set()
        for dist, i, t in pairs:
            if i in used_d or t.id in used_t or dist > 0.35 * w:
                continue
            used_d.add(i)
            used_t.add(t.id)
            t.update(dets[i], now, dt)
        for i, d in enumerate(dets):
            if i not in used_d:
                t = Track()
                t.update(d, now, dt)
                self.tracks.append(t)
        self.tracks = [t for t in self.tracks if now - t.last_seen <= LOST_GRACE]

        vis = [t for t in self.tracks if t.visible]
        if len(vis) >= 2:                       # two hands: right one is on the right of the mirror
            vis.sort(key=lambda t: t.wrist[0])
            vis[0].role, vis[-1].role = "Left", "Right"
        else:                                   # one hand: stable majority vote of MediaPipe's label
            for t in vis:
                r = sum(v == "Right" for v in t.votes)
                t.role = "Right" if r * 2 >= len(t.votes) else "Left"
        return self.tracks


# ----------------------------------------------------------------------------
# Camera filters (OpenCV)
# ----------------------------------------------------------------------------
class Filters:
    NOKIA_PAL = np.array([(15, 56, 15), (48, 98, 48), (15, 172, 139), (15, 188, 155)], np.uint8)

    def __init__(self, w, h):
        self.w, self.h = w, h
        yy, xx = np.mgrid[0:h, 0:w]
        d = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
        self.vignette = np.clip(1.15 - 0.55 * d ** 2, 0.25, 1.0).astype(np.float32)[..., None]
        self.scan = np.ones((h, 1, 1), np.float32)
        self.scan[::3] = 0.8
        self.grid = np.ones((h, w, 1), np.float32)
        self.grid[::5] = 0.85
        self.grid[:, ::5] = 0.85

    def apply(self, frame, name):
        if name == "red light":
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            g = np.clip((g - 15) * 1.2, 0, 255)
            out = np.dstack([g * 0.08, g * 0.2, g]) * 0.85 + frame.astype(np.float32) * np.array(
                [0.04, 0.08, 0.4], np.float32)
            return np.clip(out * self.vignette, 0, 255).astype(np.uint8)
        if name == "nokia":
            small = cv2.resize(frame, (self.w // 5, self.h // 5), interpolation=cv2.INTER_AREA)
            idx = np.clip(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) // 64, 0, 3)
            lcd = cv2.resize(self.NOKIA_PAL[idx], (self.w, self.h), interpolation=cv2.INTER_NEAREST)
            return (lcd.astype(np.float32) * self.grid).astype(np.uint8)
        if name == "vhs":
            f = frame.copy()
            f[:, :, 2] = np.roll(frame[:, :, 2], -7, axis=1)
            f[:, :, 0] = np.roll(frame[:, :, 0], 7, axis=1)
            f = cv2.GaussianBlur(f, (5, 1), 0).astype(np.float32)
            f = f * self.scan + np.random.normal(0, 12, f.shape[:2])[..., None]
            if np.random.rand() < 0.12:
                y = np.random.randint(0, self.h - 30)
                f[y:y + 12] = np.roll(f[y:y + 12], np.random.randint(-40, 40), axis=1)
            return np.clip(f * 1.05 + 6, 0, 255).astype(np.uint8)
        if name == "b&w":
            g = cv2.convertScaleAbs(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), alpha=1.25, beta=-20)
            return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        return frame


# ----------------------------------------------------------------------------
# Drawing helpers
# ----------------------------------------------------------------------------
def text(img, s, org, scale=0.55, color=(255, 255, 255), thick=1, center=False, shadow=True):
    (tw, th), _ = cv2.getTextSize(s, FONT, scale, thick)
    x, y = org
    if center:
        x, y = x - tw / 2, y + th / 2
    x, y = int(x), int(y)
    if shadow:
        cv2.putText(img, s, (x, y), FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), FONT, scale, color, thick, cv2.LINE_AA)


def rrect(img, x, y, w, h, r, color, alpha=1.0, thick=-1):
    """Rounded rectangle, filled or outlined, optional transparency (works on a small ROI)."""
    x, y, w, h = int(x), int(y), int(w), int(h)
    r = int(min(r, h // 2, w // 2))
    pad = 3
    x0, y0 = max(x - pad, 0), max(y - pad, 0)
    x1, y1 = min(x + w + pad, img.shape[1]), min(y + h + pad, img.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    ov = roi.copy() if alpha < 1 else roi
    lx, ly = x - x0, y - y0
    if thick < 0:
        cv2.rectangle(ov, (lx + r, ly), (lx + w - r, ly + h), color, -1)
        cv2.rectangle(ov, (lx, ly + r), (lx + w, ly + h - r), color, -1)
        for cx, cy in ((lx + r, ly + r), (lx + w - r, ly + r), (lx + r, ly + h - r), (lx + w - r, ly + h - r)):
            cv2.circle(ov, (cx, cy), r, color, -1, cv2.LINE_AA)
    else:
        cv2.line(ov, (lx + r, ly), (lx + w - r, ly), color, thick, cv2.LINE_AA)
        cv2.line(ov, (lx + r, ly + h), (lx + w - r, ly + h), color, thick, cv2.LINE_AA)
        cv2.line(ov, (lx, ly + r), (lx, ly + h - r), color, thick, cv2.LINE_AA)
        cv2.line(ov, (lx + w, ly + r), (lx + w, ly + h - r), color, thick, cv2.LINE_AA)
        for (cx, cy), a in (((lx + r, ly + r), 180), ((lx + w - r, ly + r), 270),
                            ((lx + w - r, ly + h - r), 0), ((lx + r, ly + h - r), 90)):
            cv2.ellipse(ov, (cx, cy), (r, r), a, 0, 90, color, thick, cv2.LINE_AA)
    if alpha < 1:
        cv2.addWeighted(ov, alpha, roi, 1 - alpha, 0, roi)


def lighten(c, k=90):
    return tuple(int(min(255, v + k)) for v in c)


def darken(c, k=0.55):
    return tuple(int(v * k) for v in c)


def make_strip(shots):
    """Classic vertical photo-booth strip: stacked 4:3 shots on a coloured card."""
    PW, PH = 400, 300
    pad, gap, foot = 24, 16, 110
    n = len(shots)
    W = PW + 2 * pad
    H = pad + n * PH + (n - 1) * gap + foot
    strip = np.empty((H, W, 3), np.uint8)
    strip[:] = BOOTH_BG
    for i, s in enumerate(shots):
        h, w = s.shape[:2]
        cw = min(w, int(h * 4 / 3))
        x0 = (w - cw) // 2
        crop = cv2.resize(s[:, x0:x0 + cw], (PW, PH), interpolation=cv2.INTER_AREA)
        y = pad + i * (PH + gap)
        strip[y:y + PH, pad:pad + PW] = crop
    cx, y = W // 2, H - foot + 38
    text(strip, "PHOTO BOOTH", (cx, y), 0.9, (255, 255, 255), 2, True, shadow=False)
    text(strip, datetime.now().strftime("%d %b %Y  %H:%M"), (cx, y + 38), 0.55,
         lighten(ACCENT, 40), 1, True, shadow=False)
    return strip


class Stroke:
    def __init__(self, color, pts):
        self.color, self.pts = color, pts
        self._n, self._smooth = -1, None

    def smooth(self):
        if self._n != len(self.pts):
            self._smooth = chaikin(self.pts, 2) if len(self.pts) > 2 else np.asarray(self.pts, np.float32)
            self._n = len(self.pts)
        return self._smooth


def render_strokes(art, strokes):
    """Neon glow pen: soft coloured halo, bright tube, white-hot core."""
    if not strokes:
        return
    h, w = art.shape[:2]
    # glow layer at half resolution (fast), then blurred at two radii for a soft halo
    g = np.zeros((h // 2, w // 2, 3), np.uint8)
    for s in strokes:
        p = s.smooth()
        c = s.color
        if len(p) >= 2:
            cv2.polylines(g, [(p / 2).astype(np.int32).reshape(-1, 1, 2)], False, c,
                          PEN_WIDTH // 2 + 3, cv2.LINE_AA)
        else:
            cv2.circle(g, (int(p[0][0] / 2), int(p[0][1] / 2)), PEN_WIDTH // 2 + 2, c, -1, cv2.LINE_AA)
    wide = cv2.GaussianBlur(g, (0, 0), 14).astype(np.float32)
    tight = cv2.GaussianBlur(g, (0, 0), 5).astype(np.float32)
    halo = cv2.resize(wide * 1.7 + tight * 1.1, (w, h), interpolation=cv2.INTER_LINEAR)
    art[:] = np.clip(art.astype(np.float32) + halo, 0, 255).astype(np.uint8)

    for s in strokes:
        p = s.smooth()
        c = s.color
        tube = lighten(c, 60)
        if len(p) < 2:
            ctr = (int(p[0][0]), int(p[0][1]))
            cv2.circle(art, ctr, PEN_WIDTH // 2, tube, -1, cv2.LINE_AA)
            cv2.circle(art, ctr, max(2, PEN_WIDTH // 4), (255, 255, 255), -1, cv2.LINE_AA)
            continue
        pi = p.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(art, [pi], False, tube, PEN_WIDTH, cv2.LINE_AA)                          # neon tube
        cv2.polylines(art, [pi], False, (255, 255, 255), max(2, PEN_WIDTH // 3), cv2.LINE_AA)  # white-hot core


# ----------------------------------------------------------------------------
# App
# ----------------------------------------------------------------------------
class Button:
    def __init__(self, action, label, x, y, w, h, shape="rect"):
        self.action, self.label, self.x, self.y, self.w, self.h, self.shape = action, label, x, y, w, h, shape

    def hit(self, p):
        if self.shape == "circle":
            return math.dist((self.x, self.y), p) <= self.w / 2 + 8
        return self.x - 4 <= p[0] <= self.x + self.w + 4 and self.y - 4 <= p[1] <= self.y + self.h + 4


class App:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.filters = Filters(w, h)
        self.filter_i = 0
        self.color_i = 0
        self.strokes = []
        self.cur = None            # Stroke being drawn
        self.cur_owner = None      # track id that owns it
        self.show_help = False
        self.show_skeleton = False
        self.countdown_end = None
        self.booth = None          # {"shots": [...], "t": next-shot time}
        self.strip = None          # (image, show-until) of the last finished strip
        self.flash = 0.0
        self.recording = False
        self.writer = None
        self.toast = ("", 0.0)
        self.mouse_click = None
        self.build_ui()
        os.makedirs(CAPTURE_DIR, exist_ok=True)

    # ---- layout -----------------------------------------------------------
    def build_ui(self):
        w, h = self.w, self.h
        self.buttons = []
        for i in range(len(PALETTE)):
            self.buttons.append(Button(("color", i), "", w - 52, int(h * 0.2) + i * 62, 40, 40, "circle"))
        tab_w, gap = 104, 8
        cx0 = w // 2 - 40                      # panel centre (shifted to fit the extra button)
        self.panel = (cx0 - 280, h - 142, 560, 130)
        px, py, pw, ph = self.panel
        x0 = cx0 - (len(FILTERS) * tab_w + (len(FILTERS) - 1) * gap) // 2
        for i, name in enumerate(FILTERS):
            self.buttons.append(Button(("filter", i), name, x0 + i * (tab_w + gap), py + 14, tab_w, 30))
        cy = py + 90
        c = w // 2
        self.buttons.append(Button(("undo",), "undo", c - 300, cy - 18, 75, 36))
        self.buttons.append(Button(("clear",), "clear", c - 215, cy - 18, 75, 36))
        self.buttons.append(Button(("booth",), "booth", c - 130, cy - 18, 85, 36))
        self.buttons.append(Button(("photo",), "", c, cy, 60, 60, "circle"))
        self.buttons.append(Button(("rec",), "rec", c + 50, cy - 18, 70, 36))
        self.buttons.append(Button(("help",), "how to", c + 130, cy - 18, 90, 36))

    def over_ui(self, p):
        return any(b.hit(p) for b in self.buttons)

    def press(self, action):
        k = action[0]
        if k == "color":
            self.color_i = action[1]
        elif k == "filter":
            self.filter_i = action[1]
        elif k == "undo":
            if self.strokes:
                self.strokes.pop()
                self.cur = None
        elif k == "clear":
            self.strokes.clear()
            self.cur = None
        elif k == "photo":
            if self.countdown_end is None and self.booth is None:
                self.countdown_end = time.monotonic() + 3.0
        elif k == "booth":
            if self.booth is not None:                     # press again to cancel
                self.booth = None
                self.say("Photo booth cancelled")
            elif self.countdown_end is None:
                self.booth = {"shots": [], "t": time.monotonic() + BOOTH_FIRST}
                self.strip = None
                self.say("Photo booth - strike a pose!")
        elif k == "rec":
            self.toggle_record()
        elif k == "help":
            self.show_help = not self.show_help

    def say(self, s):
        self.toast = (s, time.monotonic() + 2.5)

    def toggle_record(self):
        if self.recording:
            self.writer.release()
            self.writer, self.recording = None, False
            self.say("Video saved to captures/")
        else:
            path = os.path.join(CAPTURE_DIR, datetime.now().strftime("video_%Y%m%d_%H%M%S.mp4"))
            self.writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (self.w, self.h))
            self.recording = True
            self.say("Recording - press V or REC to stop")

    def erase(self, c, r):
        out, changed = [], False
        for s in self.strokes:
            seg, hit = [], False
            pieces = []
            for p in s.pts:
                if math.dist(p, c) < r:
                    hit = True
                    if seg:
                        pieces.append(seg)
                    seg = []
                else:
                    seg.append(p)
            if seg:
                pieces.append(seg)
            if not hit:
                out.append(s)
            else:
                changed = True
                out += [Stroke(s.color, pc) for pc in pieces]
        if changed:
            self.strokes = out
            self.cur = None

    # ---- interaction ------------------------------------------------------
    def interact(self, tracks):
        pen_active = False
        self.eraser = None
        for t in tracks:
            rising = t.stable and not t.prev_stable
            t.prev_stable = t.stable
            if not t.stable:
                t.ui_hold = False
            if rising and t.visible and self.over_ui(t.cursor):
                for b in self.buttons:
                    if b.hit(t.cursor):
                        self.press(b.action)
                        break
                t.ui_hold = True
            if t.ui_hold:
                continue

            if t.role == "Right" and t.stable:
                pen_active = True
                if self.cur is not None and self.cur_owner != t.id:
                    self.cur = None
                if self.cur is None:
                    if t.visible and t.raw and not self.over_ui(t.cursor):
                        self.cur = Stroke(PALETTE[self.color_i], [t.cursor])
                        self.cur_owner = t.id
                        self.strokes.append(self.cur)
                elif t.visible and t.raw and math.dist(self.cur.pts[-1], t.cursor) > 2.5:
                    self.cur.pts.append(t.cursor)
            elif t.role == "Left" and t.stable and t.visible and t.raw:
                self.eraser = t.cursor
                self.erase(t.cursor, ERASER_RADIUS)
        if not pen_active:
            self.cur = None

    # ---- photo booth ------------------------------------------------------
    def tick_booth(self, art, now):
        b = self.booth
        if b is None or now < b["t"]:
            return
        b["shots"].append(art.copy())            # clean frame: filter + drawing, no UI
        self.flash = 1.0
        if len(b["shots"]) >= BOOTH_SHOTS:
            strip = make_strip(b["shots"])
            path = os.path.join(CAPTURE_DIR, datetime.now().strftime("booth_%Y%m%d_%H%M%S.png"))
            cv2.imwrite(path, strip)
            self.strip = (strip, now + BOOTH_PREVIEW)
            self.booth = None
            self.say("Saved " + os.path.relpath(path, HERE))
        else:
            b["t"] = now + BOOTH_GAP

    # ---- frame ------------------------------------------------------------
    def render(self, mirrored, tracks):
        if self.mouse_click:
            for b in self.buttons:
                if b.hit(self.mouse_click):
                    self.press(b.action)
                    break
            self.mouse_click = None
        self.interact(tracks)

        art = self.filters.apply(mirrored, FILTERS[self.filter_i])
        art = art.copy() if art is mirrored else art
        render_strokes(art, self.strokes)

        now = time.monotonic()
        if self.countdown_end is not None and now >= self.countdown_end:
            self.countdown_end = None
            self.flash = 1.0
            path = os.path.join(CAPTURE_DIR, datetime.now().strftime("photo_%Y%m%d_%H%M%S.png"))
            cv2.imwrite(path, art)
            self.say("Saved " + os.path.relpath(path, HERE))
        self.tick_booth(art, now)
        if self.recording and self.writer is not None:
            self.writer.write(art)

        disp = art.copy()
        self.draw_hands(disp, tracks)
        self.draw_ui(disp, tracks, now)
        return art, disp

    def draw_hands(self, img, tracks):
        for t in tracks:
            if not t.visible:
                continue
            pts = t.points
            if self.show_skeleton:
                for a, b in HAND_LINES:
                    cv2.line(img, tuple(int(v) for v in pts[a]), tuple(int(v) for v in pts[b]),
                             (255, 255, 255), 1, cv2.LINE_AA)
            thumb = tuple(int(v) for v in pts[4])
            idx = tuple(int(v) for v in pts[8])
            cur = (int(t.cursor[0]), int(t.cursor[1]))
            if not t.raw:
                cv2.line(img, thumb, idx, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(img, thumb, 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(img, idx, 4, (255, 255, 255), -1, cv2.LINE_AA)
            if t.role == "Left":
                cv2.circle(img, cur, ERASER_RADIUS if t.raw else 18, (255, 255, 255), 2, cv2.LINE_AA)
                text(img, "eraser", (cur[0], cur[1] - (ERASER_RADIUS if t.raw else 18) - 14), 0.5, center=True)
            else:
                c = PALETTE[self.color_i]
                if t.raw:
                    cv2.circle(img, cur, PEN_WIDTH // 2 + 3, (255, 255, 255), 2, cv2.LINE_AA)
                else:
                    cv2.circle(img, cur, 18, c, 3, cv2.LINE_AA)
                    cv2.circle(img, cur, 21, (255, 255, 255), 1, cv2.LINE_AA)
                text(img, "pen", (cur[0], cur[1] - 32), 0.5, center=True)

    def draw_ui(self, img, tracks, now):
        w, h = self.w, self.h
        px, py, pw, ph = self.panel
        rrect(img, px, py, pw, ph, 26, (20, 20, 20), 0.45)
        rrect(img, px, py, pw, ph, 26, (255, 255, 255), 0.5, 1)
        rrect(img, w - 84, int(h * 0.2) - 32, 64, len(PALETTE) * 62 + 4, 32, (20, 20, 20), 0.45)
        rrect(img, w - 84, int(h * 0.2) - 32, 64, len(PALETTE) * 62 + 4, 32, (255, 255, 255), 0.5, 1)

        for b in self.buttons:
            k = b.action[0]
            if k == "color":
                sel = b.action[1] == self.color_i
                c = PALETTE[b.action[1]]
                if sel:
                    cv2.circle(img, (b.x, b.y), 25, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(img, (b.x, b.y), 18, darken(c), -1, cv2.LINE_AA)
                cv2.circle(img, (b.x, b.y), 16, c, -1, cv2.LINE_AA)
                cv2.circle(img, (b.x - 5, b.y - 5), 4, lighten(c, 110), -1, cv2.LINE_AA)
            elif k == "filter":
                sel = b.action[1] == self.filter_i
                if sel:
                    rrect(img, b.x, b.y, b.w, b.h, 15, (255, 255, 255))
                else:
                    rrect(img, b.x, b.y, b.w, b.h, 15, (255, 255, 255), 0.35, 1)
                text(img, b.label, (b.x + b.w / 2, b.y + b.h / 2), 0.5,
                     (30, 30, 30) if sel else (235, 235, 235), 1, True, shadow=False)
            elif k == "photo":
                cv2.circle(img, (b.x, b.y), 31, (255, 255, 255), 3, cv2.LINE_AA)
                cv2.circle(img, (b.x, b.y), 24, ACCENT, -1, cv2.LINE_AA)
            else:
                active = (k == "rec" and self.recording) or (k == "help" and self.show_help) \
                    or (k == "booth" and self.booth is not None)
                if active:
                    rrect(img, b.x, b.y, b.w, b.h, 18, ACCENT)
                else:
                    rrect(img, b.x, b.y, b.w, b.h, 18, (255, 255, 255), 0.35, 1)
                text(img, b.label, (b.x + b.w / 2, b.y + b.h / 2), 0.5, (255, 255, 255), 1, True, shadow=False)

        # hover ring on whatever a fingertip is over
        for t in tracks:
            if t.visible:
                for b in self.buttons:
                    if b.hit(t.cursor):
                        if b.shape == "circle":
                            cv2.circle(img, (b.x, b.y), b.w // 2 + 12, (255, 255, 255), 1, cv2.LINE_AA)
                        else:
                            rrect(img, b.x - 3, b.y - 3, b.w + 6, b.h + 6, 18, (255, 255, 255), 0.9, 1)

        # status chip (top-left)
        vis = [t for t in tracks if t.visible]
        dot = (90, 220, 120) if vis else (120, 120, 120)
        rrect(img, 16, 16, 250 if vis else 190, 34, 17, (20, 20, 20), 0.45)
        cv2.circle(img, (34, 33), 6, dot, -1, cv2.LINE_AA)
        text(img, "hand detected" if vis else "show your hand", (48, 38), 0.5, (240, 240, 240), 1, shadow=False)
        if vis:
            roles = ", ".join(sorted({("pen" if t.role == "Right" else "eraser") for t in vis}))
            text(img, "- " + roles, (150, 38), 0.45, (190, 190, 190), 1, shadow=False)
        text(img, "pinch right hand to draw  /  left hand to erase", (w // 2, h - 4 - 2), 0.42,
             (210, 210, 210), 1, True)
        if self.recording:
            cv2.circle(img, (w - 40, 33), 8, REC_RED, -1, cv2.LINE_AA)
            text(img, "REC", (w - 92, 38), 0.55)

        # photo / booth countdown
        if self.countdown_end is not None:
            n = max(1, int(math.ceil(self.countdown_end - now)))
            text(img, str(n), (w / 2, h / 2 - 20), 6, (255, 255, 255), 8, True)
        if self.booth is not None:
            n = max(1, int(math.ceil(self.booth["t"] - now)))
            text(img, str(n), (w / 2, h / 2 - 20), 6, (255, 255, 255), 8, True)
            shot_no = len(self.booth["shots"]) + 1
            text(img, f"photo {shot_no} of {BOOTH_SHOTS}", (w / 2, h / 2 - 110), 0.9,
                 (255, 255, 255), 2, True)
            for i in range(BOOTH_SHOTS):                       # progress dots
                cx = int(w / 2 + (i - (BOOTH_SHOTS - 1) / 2) * 30)
                if i < len(self.booth["shots"]):
                    cv2.circle(img, (cx, int(h / 2 + 90)), 9, ACCENT, -1, cv2.LINE_AA)
                else:
                    cv2.circle(img, (cx, int(h / 2 + 90)), 9, (255, 255, 255), 2, cv2.LINE_AA)

        # finished strip preview
        if self.strip is not None:
            if now < self.strip[1]:
                s = self.strip[0]
                sc = (h - 140) / s.shape[0]
                sm = cv2.resize(s, (int(s.shape[1] * sc), int(s.shape[0] * sc)), interpolation=cv2.INTER_AREA)
                x, y = 20, 64
                img[y:y + sm.shape[0], x:x + sm.shape[1]] = sm
                cv2.rectangle(img, (x - 2, y - 2), (x + sm.shape[1] + 2, y + sm.shape[0] + 2),
                              (255, 255, 255), 2, cv2.LINE_AA)
            else:
                self.strip = None

        if self.flash > 0:
            cv2.addWeighted(np.full_like(img, 255), self.flash, img, 1 - self.flash, 0, img)
            self.flash = max(0.0, self.flash - 0.12)
        if now < self.toast[1]:
            rrect(img, w / 2 - 190, 70, 380, 34, 17, (20, 20, 20), 0.6)
            text(img, self.toast[0], (w / 2, 87), 0.55, (255, 255, 255), 1, True, shadow=False)
        if self.show_help:
            rrect(img, w / 2 - 340, h / 2 - 200, 680, 370, 24, (15, 15, 15), 0.78)
            lines = ["HOW TO",
                     "Pinch RIGHT hand (thumb + index)   draw with colour",
                     "Pinch LEFT hand                    erase",
                     "Pinch over a colour / filter / button to pick it",
                     "Big round button: 3-2-1 photo, saved in captures/",
                     "BOOTH button: 4 photos -> a photo strip in captures/",
                     "Keys: Z undo  C clear  1-5 filters  P photo  B booth  V rec",
                     "S swap hands (if pen/eraser are reversed)  K skeleton",
                     "H closes this   Q quits"]
            for i, s in enumerate(lines):
                text(img, s, (w / 2, h / 2 - 160 + i * 38), 0.65 if i else 0.85,
                     (255, 255, 255) if i else ACCENT, 1, True)

    def key(self, k):
        if k == ord("z"):
            self.press(("undo",))
        elif k == ord("c"):
            self.press(("clear",))
        elif k == ord("p"):
            self.press(("photo",))
        elif k == ord("b"):
            self.press(("booth",))
        elif k == ord("v"):
            self.press(("rec",))
        elif k == ord("h"):
            self.press(("help",))
        elif k == ord("k"):
            self.show_skeleton = not self.show_skeleton
        elif k == ord("s"):
            SWAP_HANDS_STATE["swap"] = not SWAP_HANDS_STATE["swap"]
            for t in self.hands.tracks:
                t.votes.clear()
            self.say("Hands swapped")
        elif ord("1") <= k <= ord("5"):
            self.filter_i = min(k - ord("1"), len(FILTERS) - 1)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise SystemExit("Could not open the webcam. Try changing CAMERA_INDEX.")

    det = Detector()
    hands = HandManager()
    app = None
    win = "Glam Studio"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    prev = time.monotonic()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            if app is None:
                app = App(w, h)
                app.hands = hands

                def on_mouse(ev, x, y, flags, param):
                    if ev == cv2.EVENT_LBUTTONDOWN:
                        app.mouse_click = (x, y)
                cv2.setMouseCallback(win, on_mouse)

            now = time.monotonic()
            dt, prev = now - prev, now
            tracks = hands.update(det.detect(frame), now, dt, w)
            _, disp = app.render(frame, tracks)
            cv2.imshow(win, disp)

            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if k != 255:
                app.key(k)
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        if app is not None and app.recording:
            app.toggle_record()
        det.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()