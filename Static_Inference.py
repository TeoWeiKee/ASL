import os
import pickle
import cv2
import mediapipe as mp  # Helps isolate hand and detect 21 distinct joints
import numpy as np

MODEL_PATH = 'model.p'

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"'{MODEL_PATH}' not found. Run your static training script first.")

# Load your trained static model
with open(MODEL_PATH, 'rb') as f:
    model_dict = pickle.load(f)
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

print('Starting real-time static recognition! Show your hand to the camera.')
print('Press "Q" on your keyboard to quit.')

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract 63 hand landmarks (21 joints * x, y, z)
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])

            # Predict which letter, number, or token is shown
            prediction = model.predict([np.array(landmarks, dtype=np.float32)])
            predicted_character = prediction[0]

            cv2.putText(
                frame,
                f'Prediction: {predicted_character}',
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.3,
                (0, 255, 0),
                3,
                cv2.LINE_AA
            )
    else:
        cv2.putText(
            frame,
            'No hand detected',
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 165, 255),
            2,
            cv2.LINE_AA
        )

    cv2.imshow('ASL Real-Time Recognition (Static)', frame)

    # Press 'q' to exit the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()