import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Ask user for the gesture label
gesture_name = input("Enter gesture label: ").strip()
data = []
labels = []

# Try opening webcam at index 0
cap = cv2.VideoCapture(0)

# Fallback: if 0 fails, try 1
if not cap.isOpened():
    print("[WARNING] Camera index 0 failed, trying 1...")
    cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("[ERROR] Could not open webcam. Exiting...")
    exit()

print("[INFO] Webcam started. Show the gesture and press 'q' to stop recording.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to capture frame. Exiting...")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for lm in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
            landmarks = []
            for point in lm.landmark:
                landmarks.extend([point.x, point.y, point.z])
            data.append(landmarks)
            labels.append(gesture_name)

    cv2.putText(frame, f"Recording: {gesture_name}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("Collecting Gesture Data", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save data to .npy files
if data:
    np.save("gesture_data.npy", np.array(data))
    np.save("gesture_labels.npy", np.array(labels))
    print(f"[INFO] Collected {len(data)} samples for gesture '{gesture_name}'.")
else:
    print("[INFO] No data collected.")
