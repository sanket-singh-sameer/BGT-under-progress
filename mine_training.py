import os
import cv2
import mediapipe as mp
import numpy as np
import pickle
import absl.logging

# Suppress TensorFlow and MediaPipe logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
absl.logging.set_verbosity(absl.logging.ERROR)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Load your trained model (pickle)
with open('model.p', 'rb') as f:
    model = pickle.load(f)

# Extract and pad landmarks
def extract_hand_landmarks(results):
    try:
        landmarks = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
        if len(landmarks) < model.n_features_in_:
            landmarks += [0] * (model.n_features_in_ - len(landmarks))
        return np.array(landmarks).reshape(1, -1)
    except Exception as e:
        print(f"❌ Landmark extraction error: {e}")
        return None

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Could not open webcam.")
    exit()

while True:
    success, frame = cap.read()
    if not success or frame is None:
        print("❌ Failed to read frame.")
        break

    frame = cv2.flip(frame, 1)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw landmarks if any
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Make prediction
    input_data = extract_hand_landmarks(results)
    if input_data is not None:
        if input_data.shape[1] == model.n_features_in_:
            try:
                prediction = model.predict(input_data)[0]
                cv2.putText(frame, f"Prediction: {prediction}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            except Exception as e:
                print(f"❌ Prediction error: {e}")
        else:
            cv2.putText(frame, "Shape mismatch", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Sign Language Recognition', frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
