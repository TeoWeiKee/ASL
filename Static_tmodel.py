import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

DATA_DIR = "./Staticdata"
MODEL_PATH = "model.p"

data = []
labels = []

print("Loading static dataset...")
for label in sorted(os.listdir(DATA_DIR)):
  label_path = os.path.join(DATA_DIR, label)
  if os.path.isdir(label_path):
    for file_name in sorted(os.listdir(label_path)):
      if file_name.endswith(".npy"):
        landmarks = np.load(os.path.join(label_path, file_name))
        if landmarks.shape == (63,):
          data.append(landmarks)
          labels.append(label)
        else:
          print(
              f"Skipping {file_name} in {label}: unexpected shape"
              f" {landmarks.shape}"
          )

data = np.array(data)
labels = np.array(labels)

x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42
)

print("Training Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)
print(f"Model trained successfully! Accuracy: {score * 100:.2f}%")

with open(MODEL_PATH, "wb") as f:
  pickle.dump({"model": model}, f)

print(f"Model saved as '{MODEL_PATH}' successfully!")