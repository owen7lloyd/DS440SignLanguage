from tkinter.tix import IMAGE
import cv2
import mediapipe as mp
import glob
import math
import os
import time

# Generates landmarks.csv file for use in Model.py

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
words = []
for name in os.listdir("dataset\\SL"):
    words.append(name)


f = open("landmarks.csv", "a")
f.write("actual,")
for i in range(0,421):
  if i == 420:
    f.write(str(i))
  else:
    f.write(str(i) + ",")
f.write("\n")

for word in words:
  print(word)
  VIDEO_FILES = []
  for filename in os.listdir("dataset\\SL\\{}".format(word)):
    VIDEO_FILES.append(filename)
  with mp_hands.Hands(
      static_image_mode=True,
      max_num_hands=1,
      min_detection_confidence=0.5) as hands:

    for idx, file in enumerate(VIDEO_FILES):
      video = cv2.VideoCapture("dataset\\SL\\{}\\".format(word) + file)
      while video.isOpened():
        ret, frame = video.read()
        if not ret:
          continue
        else:
          frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
          fps = video.get(cv2.CAP_PROP_FPS)
          duration = frame_count / fps
          for k in range(0, 5):
            cv2.imshow('frame',frame)
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not results.multi_hand_landmarks:
              time.sleep(0.1)
              duration -= 0.1
            else: 
              image_height, image_width, _ = frame.shape
              landmarks = []
              for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                  landmarks.append([landmark.x, landmark.y])
                f.write(str(words.index(word)) + ",")
                for i in range(0, 21):
                  for j in range(1, 21):
                    if j > i:
                      if i == 19:
                        f.write( str(math.sqrt( (landmarks[i][0] - landmarks[j][0])**2 + (landmarks[i][1] - landmarks[j][1])**2 )))
                      else:
                        f.write( str(math.sqrt( (landmarks[i][0] - landmarks[j][0])**2 + (landmarks[i][1] - landmarks[j][1])**2 )) + "," )
                f.write("\n")
              time.sleep(math.floor(duration/5))
          video.release()
f.close()