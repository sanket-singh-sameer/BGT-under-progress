import cv2
import mediapipe as mp
from ultralytics import YOLO

# Constants
CONFIDENCE_THRESHOLD = 0.7
BOUNDING_BOX_PADDING = 20

# Initialize YOLO model
model = YOLO("yolo11x-cls.pt")  # Update with your trained model
class_names = model.names  # Get class names from the model

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Video capture setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Flip frame horizontally for mirror effect
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process hand detection
    hand_results = hands.process(rgb_frame)
    
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            # Draw hand landmarks
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style()
            )

            # Calculate bounding box with padding
            h, w = frame.shape[:2]
            x_coords = [lm.x * w for lm in hand_landmarks.landmark]
            y_coords = [lm.y * h for lm in hand_landmarks.landmark]
            
            x_min = max(0, int(min(x_coords) - BOUNDING_BOX_PADDING))
            y_min = max(0, int(min(y_coords) - BOUNDING_BOX_PADDING))
            x_max = min(w, int(max(x_coords) + BOUNDING_BOX_PADDING))
            y_max = min(h, int(max(y_coords) + BOUNDING_BOX_PADDING))

            # Extract and validate hand ROI
            hand_roi = frame[y_min:y_max, x_min:x_max]
            if hand_roi.size == 0:
                continue

            # Perform YOLO detection
            yolo_results = model(hand_roi, verbose=False)[0]  # Disable logging
            
            # Process detections
            best_conf = 0
            best_box = None
            for box in yolo_results.boxes:
                conf = box.conf.item()
                if conf > CONFIDENCE_THRESHOLD and conf > best_conf:
                    best_conf = conf
                    best_box = box

            # Draw best detection
            if best_box is not None:
                x1, y1, x2, y2 = map(int, best_box.xyxy[0])
                cls_id = int(best_box.cls.item())
                label = f"{class_names[cls_id]} {best_conf:.2f}"

                # Convert coordinates to original frame space
                abs_x1 = x_min + x1
                abs_y1 = y_min + y1
                abs_x2 = x_min + x2
                abs_y2 = y_min + y2

                # Draw bounding box and label
                cv2.rectangle(frame, (abs_x1, abs_y1), (abs_x2, abs_y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (abs_x1, abs_y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("ASL Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()