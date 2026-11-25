from collections import deque
import cv2
import mediapipe as mp
import numpy as np
import os
from tensorflow.keras.models import load_model

# Load your trained LSTM model
model = load_model('action_model.keras')
letters = sorted(os.listdir('./data'))

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Sliding window buffer to hold the last 30 frames of coordinates
window = deque(maxlen=30)
predicted_character = "Waiting..."

print('Starting real-time sequence recognition! Press "Q" to quit.')

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)
    
    current_landmarks = [0.0] * 63
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            current_landmarks = landmarks

    # Append current frame's landmarks to our rolling window
    window.append(current_landmarks)

    # Once we have a full window of 30 frames, make a prediction
    if len(window) == 30:
        res = model.predict(np.expand_dims(window, axis=0), verbose=0)[0]
        best_match_index = np.argmax(res)
        
        # Only display prediction if confidence is high (> 75%)
        if res[best_match_index] > 0.75:
            predicted_character = letters[best_match_index]

    # Display prediction on screen
    cv2.putText(frame, f'Prediction: {predicted_character}', (40, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow('ASL Dynamic Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()