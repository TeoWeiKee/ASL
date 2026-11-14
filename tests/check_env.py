import sys # System Specific Parameters and Functions => Ensure that your python run in the specific version (Python 3.12)
import numpy as np # Numerical Python => Is the array to collect the data (images, audio, and text embeddings)
import cv2 # OpenCV => A massive computer vision library used for real-time image and video processing.
import tensorflow as tf # An open-source develop by Google => A tools that trains model by running millions of calculation until the model learn the pattern.
# Tensor (Data) = All data are stored and manipualated as tensors.
# Flow (Computation) = Start comparing the data (through network) to get closer with the correct answer. (Which call training)
import sklearn # Scikit Learn => Data preprocessing and evaluation (data processing , metrics calculation)
# Tells Python open and load libraries
# (OpenCV for Video, TensorFlow for machine learning, NumPy for math)

print("Python:", sys.version.split()[0])
print("NumPy:", np.__version__) # Check NumPy version
print("OpenCV:", cv2.__version__) # Check OpenCV Version
print("TensorFlow:", tf.__version__) # Check TensorFlow Version
print("scikit-learn:", sklearn.__version__) # Check Scikit Learn Version
print("GPUs found:", tf.config.list_physical_devices("GPU")) # Check TensorFlow can activate GPU to speed out training or not

cap = cv2.VideoCapture(0) # OpenCV connect the computer primary webcam
ok, frame = cap.read() # Snap single picture if ok
cap.release()
print("Webcam works:", ok, "| frame shape:", frame.shape if ok else None) # Print the picture if ok