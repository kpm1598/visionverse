
import cv2

# Open the default webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam started! Press Q to quit.")

while True:
    # Capture one frame from the webcam
    success, frame = camera.read()

    if not success:
        print("Could not read webcam frame.")
        break

    # Mirror the frame like a normal mirror
    frame = cv2.flip(frame, 1)

    # Display the live video
    cv2.imshow("Air Arcade - Webcam Test", frame)

    # Quit when Q is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
