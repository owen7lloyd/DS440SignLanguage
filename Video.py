from tkinter.tix import IMAGE
import cv2
import mediapipe as mp
import math
import os

# Generates landmarks.csv file for use in Model.py

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
directory = "alphabet_video"

def find_hand_period(path):
  video = cv2.VideoCapture(path)
  start = None
  end = None
  counter = 0
  last_frame = None
  hands_found = False

  while video.isOpened():
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5) as hands:  

      while hands_found == False:
        last_frame = counter
        video.set(cv2.CAP_PROP_POS_FRAMES, counter)
        _, frame = video.read()
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if counter >= video.get(cv2.CAP_PROP_FRAME_COUNT) - 14:
          start = 0
          end = 0
          hands_found = True
        else:
          if results.multi_hand_landmarks:
            hands_found = True
          counter += 12

      while start == None:
        last_frame = counter
        video.set(cv2.CAP_PROP_POS_FRAMES, counter)
        _, frame = video.read()
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not results.multi_hand_landmarks:
          start = last_frame
          counter += 4
        else:
          if counter == 0:
            start = 0
          else:
            counter -= 4

      last_frame = counter
      video.set(cv2.CAP_PROP_POS_FRAMES, counter)
      _, frame = video.read()
      results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
      if results.multi_hand_landmarks:
        if counter >= video.get(cv2.CAP_PROP_FRAME_COUNT) - 6:
          end = counter
          video.release()
        else:
          counter += 4
      else:
        end = last_frame
        video.release()

  return start, end

def get_word_list():
  words = []
  for name in os.listdir(directory):
      words.append(name)
  return words

def get_paths(word):
  paths = []
  for filename in os.listdir("{}\\{}".format(directory, word)):
    paths.append(filename)
  return paths

def videos_to_csv(word_list):
  # Open file
  f = open("landmarks.csv", "a")

  # Write first line of csv (column names)
  f.write("actual,")
  for i in range(0,844):
    if i == 843:
      f.write(str(i))
    else:
      f.write(str(i) + ",")
  f.write("\n")


  for word in word_list:
    percent = round( (word_list.index(word) / len(word_list) ) * 100, 2)
    os.system('cls')
    print("Progress: {}%".format(percent))
    paths = get_paths(word) # Add all file paths to list
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5) as hands:
        for _, file in enumerate(paths):
          path = "{}\\{}\\".format(directory, word) + file
          video = cv2.VideoCapture(path)

          while video.isOpened():
            frame_id = 0
            collected_frames = 0
            start, end = find_hand_period(path)
            mesh_dur = end - start      # Length of time that hand mesh is present
            frame_skip = (mesh_dur - 1) / 4   # Number of frames to skip ahead in hand motion (higher number = more samples but takes longer)
            frame_id += start           # Position in video

            if start == None and end == None:
              continue
            else:
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
                  if collected_frames == 0:
                    f.write( str(landmarks[3][0]) + "," + str(landmarks[3][1]) + ","+ str(landmarks[7][0]) + "," + str(landmarks[7][1]) + ",")
                  
                  # Write to file the index of the word as a classifier followed by distance calcs
                  for i in range(0, 21):
                    for j in range(1, 21):
                      distance = math.sqrt( (landmarks[i][0] - landmarks[j][0])**2 + (landmarks[i][1] - landmarks[j][1])**2 )
                      if j > i:
                        if i == 19:
                          f.write( str(distance))
                        else:
                          f.write( str(distance) + "," )
                  collected_frames += 1
                  frame_id += frame_skip
              video.release()
        f.write("\n")
  f.close()

# Need commented out if running Main.py otherwise this will run when Main does
word_list = get_word_list()
videos_to_csv(word_list)
