import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

DATA_DIR = './data'
letters = sorted(os.listdir(DATA_DIR))

# Map each letter label to a number index
label_map = {label: num for num, label in enumerate(letters)}

sequences, labels = [], []

print("Loading sequence data...")
for letter in letters:
    letter_path = os.path.join(DATA_DIR, letter)
    if os.path.isdir(letter_path):
        for file_name in os.listdir(letter_path):
            if file_name.endswith('.npy'):
                window = np.load(os.path.join(letter_path, file_name))
                sequences.append(window)
                labels.append(label_map[letter])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

print("Building LSTM model architecture...")
model = Sequential([
    LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 63)),
    Dropout(0.2),
    LSTM(128, return_sequences=False, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(len(letters), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Training model...")
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# Save the trained deep learning model
model.save('action_model.keras')
print("Model trained and saved successfully as 'action_model.keras'!")