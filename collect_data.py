import os
import cv2
import mediapipe as mp
import numpy as np

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Define your classes (you can include letters like J and Z here)
letters = ['A', 'B', 'C', 'J', 'Z'] 

no_sequences = 30       # Number of recorded video samples per letter
sequence_length = 30    # Number of frames captured in each sequence

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

for letter in letters:
    letter_dir = os.path.join(DATA_DIR, letter)
    if not os.path.exists(letter_dir):
        os.makedirs(letter_dir)
        
    print(f'=== Collecting sequences for: {letter} ===')
    print('Press "Q" on your keyboard when you are ready to start recording.')
    
    # Wait until you press 'q' to start recording sequences for this letter
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f'Ready for "{letter}"? Press "Q" to start', (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow('Data Collection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Record multiple sequences
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            success, frame = cap.read()
            if not success:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # Extract 63 coordinates (21 joints * 3 axes)
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])
                    window.append(landmarks)
            else:
                # Fallback if hand drops out for a frame
                window.append([0.0] * 63)

            cv2.putText(frame, f'Letter: {letter} | Seq: {sequence+1}/{no_sequences} | Frame: {frame_num+1}/{sequence_length}', 
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow('Data Collection', frame)
            cv2.waitKey(1)
            
        # Save the full motion sequence as a single .npy file
        npy_path = os.path.join(letter_dir, f'{sequence}.npy')
        np.save(npy_path, np.array(window))

cap.release()
cv2.destroyAllWindows()
print("Sequence collection completed successfully!")