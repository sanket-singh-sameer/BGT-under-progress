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

# Load your trained model and metadata (pickle)
with open('model.p', 'rb') as f:
    data = pickle.load(f)
    model = data['model']  # actual classifier
    labels = data.get('labels', None)  # optional label mapping

# Extract and pad landmarks
def extract_hand_landmarks(results):
    landmarks = []
    try:
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
        if len(landmarks) < model.n_features_in_:
            landmarks += [0] * (model.n_features_in_ - len(landmarks))
        return np.array(landmarks).reshape(1, -1)
    except Exception as e:
        print(f"\u274c Landmark extraction error: {e}")
        return None

# Initialize webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    # Flip frame for mirror effect
    frame = cv2.flip(frame, 1)

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Make prediction
    if results.multi_hand_landmarks:
        input_data = extract_hand_landmarks(results)
        if input_data is not None and input_data.shape[1] == model.n_features_in_:
            prediction_index = model.predict(input_data)[0]
            label = labels[prediction_index] if labels else prediction_index
            cv2.putText(frame, f"Prediction: {label}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Input shape mismatch", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Sign Language Recognition', frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()