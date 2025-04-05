import cv2
import torch
import mediapipe as mp
from ultralytics import YOLO

# Load the YOLO model
model_path = "yolo11n.pt"  # Update with your trained model
model = YOLO(model_path)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ASL class labels (ensure these match your YOLO model's training)
CLASS_LABELS = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E",
    5: "F", 6: "G", 7: "H", 8: "I", 9: "J",
    10: "K", 11: "L", 12: "M", 13: "N", 14: "O",
    15: "P", 16: "Q", 17: "R", 18: "S", 19: "T",
    20: "U", 21: "V", 22: "W", 23: "X", 24: "Y", 25: "Z"
}

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get hand bounding box with boundary checks
            h, w, _ = frame.shape
            x_min = max(0, int(min([lm.x for lm in hand_landmarks.landmark]) * w))
            y_min = max(0, int(min([lm.y for lm in hand_landmarks.landmark]) * h))
            x_max = min(w, int(max([lm.x for lm in hand_landmarks.landmark]) * w))
            y_max = min(h, int(max([lm.y for lm in hand_landmarks.landmark]) * h))

            # Check valid ROI dimensions
            if x_max <= x_min or y_max <= y_min:
                continue

            # Crop and preprocess hand region
            hand_roi = frame[y_min:y_max, x_min:x_max]
            if hand_roi.size == 0:
                continue

            # Convert to RGB and resize for YOLO
            hand_roi_rgb = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2RGB)
            
            # Run YOLO inference
            yolo_results = model(hand_roi_rgb, verbose=False)

            # Process detections
            for r in yolo_results:
                boxes = r.boxes
                for box in boxes:
                    # Filter low confidence detections
                    if box.conf.item() < 0.5:
                        continue
                        
                    # Convert coordinates to original frame space
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_id = int(box.cls.item())
                    
                    # Draw on original frame
                    cv2.rectangle(frame, 
                                 (x_min + x1, y_min + y1),
                                 (x_min + x2, y_min + y2),
                                 (0, 255, 0), 2)
                    
                    label = f"{CLASS_LABELS.get(cls_id, '?')} {box.conf.item():.2f}"
                    cv2.putText(frame, label,
                              (x_min + x1, y_min + y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                              (0, 255, 0), 2)

    cv2.imshow("ASL Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()