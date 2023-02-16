from tkinter.tix import IMAGE
import cv2
import mediapipe as mp
import glob
import math

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
alphabet = ['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']

f = open("Hands\\landmarks.csv", "a")
f.write("actual,")
for i in range(0,421):
  if i == 420:
    f.write(str(i))
  else:
    f.write(str(i) + ",")
f.write("\n")

for letter in alphabet:
  IMAGE_FILES = []
  for filename in glob.iglob("C:\\Users\\Caden\\Desktop\\Hands\\asl_train\\{}\\".format(letter) + '*.png', recursive=True):
    IMAGE_FILES.append(filename)
  for filename in glob.iglob("C:\\Users\\Caden\\Desktop\\Hands\\asl_train\\{}\\".format(letter) + '*.jpg', recursive=True):
    IMAGE_FILES.append(filename)
  with mp_hands.Hands(
      static_image_mode=True,
      max_num_hands=1,
      min_detection_confidence=0.5) as hands:

    for idx, file in enumerate(IMAGE_FILES):
      image = cv2.flip(cv2.imread(file), 1)
      results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
      if not results.multi_hand_landmarks:
        continue

      image_height, image_width, _ = image.shape
      landmarks = []
      for hand_landmarks in results.multi_hand_landmarks:
        for landmark in hand_landmarks.landmark:
          landmarks.append([landmark.x, landmark.y])
      f.write(str(alphabet.index(letter)) + ",")
      
      for i in range(0, 21):
        for j in range(1,21):
          if j > i:
            if i == 19:
              f.write( str(math.sqrt( (landmarks[i][0] - landmarks[j][0])**2 + (landmarks[i][1] - landmarks[j][1])**2 )) + "," )
            else:
              f.write( str(math.sqrt( (landmarks[i][0] - landmarks[j][0])**2 + (landmarks[i][1] - landmarks[j][1])**2 )) + "," )
      f.write("\n")
f.close()