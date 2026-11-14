import pickle
import cv2
import mediapipe as mp # Help to isolates hand and detect 21 distinct joints
import numpy as np

# Load your trained model
model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

print('Starting real-time recognition! Show your hand to the camera.')
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

            # Extract hand landmarks
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])

            # Predict which letter you are making
            prediction = model.predict([np.array(landmarks)])
            predicted_character = prediction[0]

            # Display the predicted letter on the screen in big green text
            cv2.putText(frame, f'Prediction: {predicted_character}', (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow('ASL Real-Time Recognition', frame)

    # Press 'q' to exit the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()