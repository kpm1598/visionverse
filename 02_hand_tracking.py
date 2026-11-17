
import os

import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Path to the AI model (it sits next to this script, in the repo root)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

# Configure the hand-tracking model
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1
)

# Create the detector
detector = vision.HandLandmarker.create_from_options(options)

# Open webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Could not open webcam.")
    detector.close()
    raise SystemExit

print("Hand tracker running. Press Q to quit.")

timestamp_ms = 0

while True:
    success, frame = camera.read()

    if not success:
        print("Could not read webcam frame.")
        break

    # Mirror the image
    frame = cv2.flip(frame, 1)

    # Convert OpenCV BGR image to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Wrap the image for MediaPipe
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # VIDEO mode requires increasing timestamps
    timestamp_ms += 33

    # Detect hand landmarks
    result = detector.detect_for_video(mp_image, timestamp_ms)

    height, width, _ = frame.shape

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]

        # Draw all 21 landmarks
        for landmark in hand:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

        # Landmark 8 = index fingertip
        tip = hand[8]
        tip_x = int(tip.x * width)
        tip_y = int(tip.y * height)

        # Highlight the fingertip
        cv2.circle(frame, (tip_x, tip_y), 12, (0, 0, 255), -1)
        cv2.putText(
            frame,
            "Index fingertip",
            (tip_x + 12, tip_y - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

    cv2.imshow("Air Arcade - AI Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
detector.close()
cv2.destroyAllWindows()
