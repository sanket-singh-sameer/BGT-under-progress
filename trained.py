import cv2
import mediapipe as mp
import numpy as np
import pickle

# Load model
with open("model.p", "rb") as f:
    model = pickle.load(f)

expected_len = model.n_features_in_
print(f"[INFO] Model expects {expected_len} features")

# Setup MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)  # use 0 or 1 based on your system

with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("❌ Camera error")
            break

        H, W, _ = frame.shape
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        data_aux = []
        x_, y_ = [], []

        if results.multi_hand_landmarks:
            # Collect all hand landmarks into one array (both hands)
            for hand_landmarks in results.multi_hand_landmarks[:2]:  # just in case more than 2 detected
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                for lm in hand_landmarks.landmark[:21]:
                    x, y, z = lm.x, lm.y, lm.z
                    data_aux.extend([x, y, z])
                    x_.append(x)
                    y_.append(y)

            # Handle feature length mismatch
            actual_len = len(data_aux)
            print(f"[DEBUG] Collected {actual_len} features")

            # Pad or truncate to match expected size
            if actual_len < expected_len:
                data_aux.extend([0.0] * (expected_len - actual_len))
            elif actual_len > expected_len:
                data_aux = data_aux[:expected_len]

            try:
                prediction = model.predict([data_aux])[0]

                x1, y1 = int(min(x_) * W) - 20, int(min(y_) * H) - 20
                x2, y2 = int(max(x_) * W) + 20, int(max(y_) * H) + 20
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 99, 173), 4)
                cv2.putText(frame, prediction, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

            except Exception as e:
                print(f"❌ Prediction error: {e}")

        cv2.imshow("Hand Sign Recognition", frame)

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
