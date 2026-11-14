import os
import cv2
import mediapipe as mp
import numpy as np

# 1. Set up directories to save data
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Define all the letters and special tokens you want to collect
letters = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
    'U', 'V', 'W', 'X', 'Y', 'Z', 'nothing', 'space'
]

dataset_size = 100  # Number of samples to collect per letter/class

# Initialize MediaPipe Hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

for letter in letters:
    # Create a folder for each letter/class if it doesn't exist
    letter_dir = os.path.join(DATA_DIR, letter)
    if not os.path.exists(letter_dir):
        os.makedirs(letter_dir)
        
    print(f'=== Collecting data for: {letter} ===')
    print('Press "Q" when you are ready to start recording this sign.')
    
    # Wait until user presses 'q' to start recording that sign
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f'Ready for "{letter}"? Press "Q" to start', (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow('Data Collection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        success, frame = cap.read()
        if not success:
            break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Extract landmarks
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
                
                # Save landmarks to a file
                np.save(os.path.join(letter_dir, f'{counter}.npy'), np.array(landmarks))
                counter += 1
                
                # Display progress on screen
                cv2.putText(frame, f'Collected: {counter}/{dataset_size}', (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Data Collection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("Full dataset collection completed successfully!")