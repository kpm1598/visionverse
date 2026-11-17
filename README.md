# VisionVerse

Webcam + AI hand-tracking experiments built with **OpenCV** and **MediaPipe**
(Google's Hand Landmarker model). Draw in the air with a pinch, apply live
camera filters, and take photo-booth strips — no mouse or touchscreen needed.

## What's inside

| File | What it does |
| --- | --- |
| `01_webcam_test.py` | Step 1 — opens the webcam, shows the mirrored live feed. Press **Q** to quit. |
| `02_hand_tracking.py` | Step 2 — runs the Hand Landmarker on the webcam feed and draws all 21 hand landmarks, highlighting the index fingertip. Press **Q** to quit. |
| `pinch_pop.py` | The main app — **"Glam Studio"**: air-drawing, filters and a photo booth (see controls below). |
| `hand_landmarker.task` | The MediaPipe hand-tracking model used by 02 and pinch_pop. `pinch_pop.py` re-downloads it automatically if it's missing. |
| `captures/` | Photos, photo-booth strips and recordings saved by the apps land here. |

## Setup (Windows)

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell refuses to run `Activate.ps1` ("running scripts is disabled"),
skip activation and call the venv's Python directly:

```powershell
.venv\Scripts\python.exe pinch_pop.py
```

## Run

Run the steps in order, or jump straight to the main app:

```powershell
python 01_webcam_test.py
python 02_hand_tracking.py
python pinch_pop.py
```

You need a working webcam. If the camera doesn't open, close any other app
using it, or change `CAMERA_INDEX` at the top of the script.

## Glam Studio controls (pinch_pop.py)

- **Pinch right hand** (thumb + index together) — draw with the neon pen
- **Pinch left hand** — erase
- **Pinch over a colour, filter or button** on screen to select it
- **Big round button** — 3-2-1 photo, saved to `captures/`
- **BOOTH button** — 4 photos turned into a photo-booth strip, saved to `captures/`

Keyboard:

| Key | Action |
| --- | --- |
| Z | Undo last stroke |
| C | Clear drawing |
| 1–5 | Switch filter (normal, red light, nokia, vhs, b&w) |
| P | Photo |
| B | Photo booth |
| V | Start/stop video recording |
| S | Swap hands (if pen/eraser feel reversed) |
| K | Show/hide hand skeleton |
| H | Help overlay |
| Q / Esc | Quit |

## Notes

- Everything runs locally on your machine — no data leaves the computer,
  except the one-time model download in `pinch_pop.py` if
  `hand_landmarker.task` is missing.
- Versions in `requirements.txt` are pinned to the set verified on
  17 Nov 2026 (Windows wheels confirmed for all three packages).
