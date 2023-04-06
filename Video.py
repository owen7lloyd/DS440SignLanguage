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
  start = None
  end = None
  counter = 0
  last_frame = None

  while video.isOpened():
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5) as hands:
      video.set(cv2.CAP_PROP_POS_FRAMES, counter)
      _, frame = video.read()
      results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
      if counter >= video.get(cv2.CAP_PROP_FRAME_COUNT) - 4:
        if start != None:
          end = last_frame
          video.release()
        else:
          start, end = None
          video.release()
      if results.multi_hand_landmarks:
        if start == None:
          start = counter
        else:
          pass
      else:
        if start != None:
          end = last_frame
          video.release()
        else:
          pass
      last_frame = counter
      counter += 2
  return start, end

def get_word_list():
  words = []
  for name in os.listdir("data_sample"):
      words.append(name)
  return words

def get_paths(word):
  paths = []
  for filename in os.listdir("data_sample\\{}".format(word)):
    paths.append(filename)
  return paths

def videos_to_csv(word_list):
  # Open file
  f = open("landmarks.csv", "a")

  # Write first line of csv (column names)
  f.write("actual,")
  for i in range(0,1679):
    if i == 1678:
      f.write(str(i))
    else:
      f.write(str(i) + ",")
  f.write("\n")


  for word in word_list:
    percent = round( (word_list.index(word) / len(word_list) ) * 100, 2)
    print("Progress: {}%".format(percent))
    paths = get_paths(word)
    # Add all file paths to list
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5) as hands:

      for idx, file in enumerate(paths):
        path = "data_sample\\{}\\".format(word) + file
        video = cv2.VideoCapture(path)
        while video.isOpened():
          frame_id = 0
          collected_frames = 0

          start, end = find_hand_period(path)
          if start == None and end == None:
            continue
          else:
            mesh_dur = end - start      # Length of time that hand mesh is present
            frame_skip = mesh_dur / 4   # Number of frames to skip ahead in hand motion (higher number = more samples but takes longer)
            frame_id += start           # Position in video
            prev_landmarks = []

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

                f.write(str(word_list.index(word)) + ",")      
                # Write to file the index of the word as a classifier followed by distance calcs
                for i in range(0, 21):
                  difference = [math.sqrt( (landmarks[i][0] - x[0])**2 + (landmarks[i][1] - x[1])**2 ) for x in prev_landmarks]
                  if prev_landmarks != []:
                          for dif in difference:
                            f.write(str(dif) + ",")
                  for j in range(1, 21):
                    distance = math.sqrt( (landmarks[i][0] - landmarks[j][0])**2 + (landmarks[i][1] - landmarks[j][1])**2 )
                    if j > i:
                      if i == 19:
                        f.write( str(distance))
                      else:
                        f.write( str(distance) + "," )
                collected_frames += 1
                frame_id += frame_skip
              prev_landmarks = landmarks
            video.release()
        f.write("\n")
  f.close()

# Need commented out if running Main.py otherwise this will run when Main does
word_list = get_word_list()
videos_to_csv(word_list)