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


def find_hand_period(word, file):
  video = cv2.VideoCapture("dataset\\SL\\{}\\".format(word) + file)
  start = 0
  end = 0
  counter = 0
  last_frame = 0

  while video.isOpened():
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5) as hands:
      video.set(cv2.CAP_PROP_POS_FRAMES, counter)
      ret, frame = video.read()
      results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
      if counter == video.get(cv2.CAP_PROP_FRAME_COUNT) - 2:
        end = last_frame
        video.release()
      elif not results.multi_hand_landmarks:
        if last_frame > 0:
          end = last_frame
          video.release()
        counter += 1
      elif results.multi_hand_landmarks and (start == 0 or end == 0):
        if start == 0:
          start = counter
        else:
          pass
        last_frame = counter
      else: # When both start and end have a value other than 0
        break
      counter += 1
  return start, end

def get_word_list():
  words = []
  for name in os.listdir("dataset\\SL"):
      words.append(name)
  return words

def write_data():
  words = get_word_list()
  # Open file
  f = open("landmarks.csv", "a")
  # Write first line of csv (column names)
  f.write("actual,")
  for i in range(0,1684):
    if i == 1683:
      f.write(str(i))
    else:
      f.write(str(i) + ",")
  f.write("\n")


  for word in words:
    print(word)
    # Add all file paths to list
    VIDEO_FILES = []
    for filename in os.listdir("dataset\\SL\\{}".format(word)):
      VIDEO_FILES.append(filename)
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5) as hands:

      for idx, file in enumerate(VIDEO_FILES):
        print(file)
        video = cv2.VideoCapture("dataset\\SL\\{}\\".format(word) + file)
        while video.isOpened():
          counter = 0
          frame_id = 0
          collected_frames = 0

          start, end = find_hand_period(word, file)
          mesh_dur = end - start
          frame_skip = mesh_dur / 4
          frame_id += start

          while collected_frames < 4:
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = video.read()
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not results.multi_hand_landmarks:
              counter += 0.5
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
              collected_frames += 1
              frame_id += frame_skip
          video.release()
        f.write("\n")
  f.close()

write_data()