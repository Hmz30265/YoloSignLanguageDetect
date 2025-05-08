import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
import mediapipe as mp
import time

model = YOLO("dataAugmented_runs/detect/train2/weights/best.pt")
mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1)

LETTER_CONFIDENCE = 0.7 
VOTING_WINDOW = 3 

current_letter_predictions = defaultdict(list)
last_letter_time = None
current_word = []
display_letter = None
is_new_letter = True
current_boxes = []

def get_best_letter():
    if not current_letter_predictions:
        return None

    letter_stats = {
        letter: (len(confs), np.mean(confs))
        for letter, confs in current_letter_predictions.items()
        if confs
    }

    if not letter_stats:
        return None

    best_letter = max(letter_stats.items(), key=lambda item: (item[1][0], item[1][1]))[0]
    return best_letter

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    current_boxes = []
  
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = mp_hands.process(rgb_frame)
    
    # Process frame if hands are detected
    if hand_results.multi_hand_landmarks:
        if is_new_letter:
            last_letter_time = time.time()
            is_new_letter = False
            display_letter = None
            current_letter_predictions.clear()
        
        yolo_results = model(frame, verbose=False)[0]
        
        # Collect all predictions above threshold
        for box, cls, conf in zip(yolo_results.boxes.xyxy, yolo_results.boxes.cls, yolo_results.boxes.conf):
            if conf > LETTER_CONFIDENCE:
                letter = model.names[int(cls)]
                current_letter_predictions[letter].append(float(conf))
                current_boxes.append((box, letter, float(conf)))
    
        for box, letter, conf in current_boxes:
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            cv2.putText(frame, f"{letter} {conf:.2f}", (int(box[0]), int(box[1])-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Check if voting period is complete
        elapsed = time.time() - last_letter_time
        if elapsed >= VOTING_WINDOW:
            best_letter = get_best_letter()
            if best_letter:
                current_word.append(best_letter)
                display_letter = best_letter
            is_new_letter = True
            
        time_remaining = max(0, VOTING_WINDOW - elapsed)
    else:
        if current_word:
            current_word = []
        is_new_letter = True
        display_letter = None
        time_remaining = 0
    
    if display_letter:
        cv2.putText(frame, f"Identified: {display_letter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        status_text = f"Collecting... {time_remaining:.1f}s" if time_remaining > 0 else "Show your hand"
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    cv2.putText(frame, f"Current Word: {''.join(current_word)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    cv2.imshow("ASL Fingerspelling", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()