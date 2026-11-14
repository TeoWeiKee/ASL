import cv2

# 1. Turn on the webcam
cap = cv2.VideoCapture(0)

# 2. Capture a single picture (frame)
ok, frame = cap.read()

if ok:
  print("Picture successfully captured!")

  # Option A: Save the picture to your project folder
  cv2.imwrite("my_first_photo.jpg", frame)
  print("Saved as 'my_first_photo.jpg' in your folder!")

  # Option B: Pop up a window to display the picture on your screen
  cv2.imshow("Captured Photo Test", frame)

  print(
      "Look at the new window that popped up! Press ANY key on your keyboard"
      " to close it."
  )
  cv2.waitKey(0)  # This pauses the code until you press a key
  cv2.destroyAllWindows()  # Close the image window
else:
  print("Oops! Could not read frame from camera.")

# 3. Turn off the webcam
cap.release()