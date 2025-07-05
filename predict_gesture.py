import cv2
import mediapipe as mp
import numpy as np
import pickle

# Load model
with open("gesture_model.pkl", "rb") as f:
    model, encoder = pickle.load(f)

# Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)
print("[INFO] Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for lm in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
            landmarks = []
            for point in lm.landmark:
                landmarks.extend([point.x, point.y, point.z])
            if len(landmarks) == 63:
                prediction = model.predict([landmarks])[0]
                gesture_name = encoder.inverse_transform([prediction])[0]
                cv2.putText(frame, f"Gesture: {gesture_name}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
