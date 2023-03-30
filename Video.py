from tkinter.tix import IMAGE
import cv2
import mediapipe as mp
import math
import os

# Generates landmarks.csv file for use in Model.py

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


def find_hand_period(path):
  video = cv2.VideoCapture(path)
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
      if counter >= video.get(cv2.CAP_PROP_FRAME_COUNT) - 6:
        end = last_frame
        video.release()
      elif not results.multi_hand_landmarks:
        if last_frame > 0:
          end = last_frame
          video.release()
        counter += 2
      elif results.multi_hand_landmarks and (start == 0 or end == 0):
        if start == 0:
          start = counter
        else:
          pass
        last_frame = counter
      else: # When both start and end have a value other than 0
        break
      counter += 2
  return start, end

def get_word_list():
  words = []
  for name in os.listdir("dataset\\SL"):
      words.append(name)
  return words

def get_paths(word):
  paths = []
  for filename in os.listdir("dataset\\SL\\{}".format(word)):
    paths.append(filename)
  return paths

def videos_to_csv(word_list):
  # Open file
  f = open("landmarks.csv", "a")

  # Write first line of csv (column names)
  f.write("actual,")
  for i in range(0,840):
    if i == 839:
      f.write(str(i))
    else:
      f.write(str(i) + ",")
  f.write("\n")


  for word in word_list:
    print(word)
    paths = get_paths(word)
    # Add all file paths to list
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5) as hands:

      for idx, file in enumerate(paths):
        print(file)
        path = "dataset\\SL\\{}\\".format(word) + file
        video = cv2.VideoCapture(path)
        while video.isOpened():
          frame_id = 0
          collected_frames = 0

          start, end = find_hand_period(path)
          mesh_dur = end - start      # Length of time that hand mesh is present
          frame_skip = mesh_dur / 4   # Number of frames to skip ahead in hand motion (higher number = more samples but takes longer)
          frame_id += start           # Position in video

          while collected_frames < 4:
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            _, frame = video.read()
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not results.multi_hand_landmarks:
              frame_id += 1    # Skip to next frame if hands are not present
            else: 
              landmarks = []
              for landmark in results.multi_hand_landmarks[0].landmark: # Add x,y coords of hand mesh points to list as tuples
                landmarks.append([landmark.x, landmark.y]) 
              f.write(str(word_list.index(word)) + ",")               # Write to file the index of the word as a classifier followed by distance calcs
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

# Need commented out if running Main.py otherwise this will run
word_list = get_word_list()
videos_to_csv(word_list)