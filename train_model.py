import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATA_DIR = './data'

data = []
labels = []

# Load all recorded landmark files from the data folders
print("Loading dataset...")
for letter in os.listdir(DATA_DIR):
    letter_path = os.path.join(DATA_DIR, letter)
    if os.path.isdir(letter_path):
        for img_file in os.listdir(letter_path):
            if img_file.endswith('.npy'):
                landmarks = np.load(os.path.join(letter_path, img_file))
                data.append(landmarks)
                labels.append(letter)

data = np.array(data)
labels = np.array(labels)

# Split data into training and testing sets (80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, random_state=42)

# Train a fast and accurate Random Forest classifier
print("Training model...")
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Test model accuracy
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)
print(f"Model trained successfully! Accuracy: {score * 100:.2f}%")

# Save the trained model to a file
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

print("Model saved as 'model.p' successfully!")