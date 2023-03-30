import os
import cv2
import numpy as np
from PIL import Image
from keras import models
import tensorflow as tf
import mediapipe as mp
import math
from Video import find_hand_period

# Runs real-time ASL prediction

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Used for single image prediction #
def predict_image():
    model = models.load_model('Models\\model.h5')
    guesses = ['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y'] 
  #  url = "https://www.signingsavvy.com/images/words/alphabet/2/u1.jpg"
  #  path = tf.keras.utils.get_file('u', url)
    path = "asl_alphabet_test\\S_test.jpg"
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5) as hands:
        image = cv2.flip(cv2.imread(path), 1)
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  
        
    landmarks = []
    for hand_landmarks in results.multi_hand_landmarks:
        for landmark in hand_landmarks.landmark:
            landmarks.append((landmark.x, landmark.y))
    result = []
    for i in range(0, 21):
        for j in range(1,21):
            if j > i:
                result.append(math.sqrt( abs(landmarks[i][0] - landmarks[j][0])**2 + abs(landmarks[i][1] - landmarks[j][1])**2 ))
    result = np.array([result])
    predictions = model.predict(result)
    score = tf.nn.softmax(predictions[0])
    scores = list(zip(guesses, list(predictions[0])))
    pred_list = []
    for tup in scores:
        pred_list += [tup[1]]
    confidence = max(pred_list) * 100

    print("This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(guesses[np.argmax(score)], confidence)
    )

def predict_video(video):
    model = models.load_model('Models\\model2.h5')
    words = []
    for name in os.listdir("alphabet_video"):
        words.append(name)
    result = []
    vid = cv2.VideoCapture(video)

    while vid.isOpened():
        frame_id = 0
        collected_frames = 0

        start, end = find_hand_period(video)
        mesh_dur = end - start
        frame_skip = mesh_dur / 4
        frame_id += start

        with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5) as hands:
            while collected_frames < 4:
                vid.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                _, frame = vid.read()
                results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if not results.multi_hand_landmarks:
                    frame_id += 1
                else: 
                    landmarks = []
                    for landmark in results.multi_hand_landmarks[0].landmark:
                        landmarks.append([landmark.x, landmark.y])
                    for i in range(0, 21):
                        for j in range(1, 21):
                            if j > i:
                                result.append(math.sqrt( (landmarks[i][0] - landmarks[j][0])**2 + (landmarks[i][1] - landmarks[j][1])**2 ))
                                result.append(math.atan( abs(landmarks[i][0] - landmarks[j][0]) / abs(landmarks[i][1] - landmarks[j][1])))

                    collected_frames += 1
                    frame_id += frame_skip
            vid.release()

    result = np.array([result])
    predictions = model.predict(result)
    score = tf.nn.softmax(predictions[0])
    scores = list(zip(words, list(predictions[0])))
    pred_list = []
    for tup in scores:
        pred_list += [tup[1]]
    confidence = max(pred_list) * 100

    print("This video most likely portrays \"{}\" with a {:.2f}% confidence."
        .format(words[np.argmax(score)], confidence)
    )

# Used for webcam prediction (alphabet_only)
def live_prediction():
    model = models.load_model('Models\\model.h5')
    guesses = ['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        path2 = "asl_alphabet_test\\"
        _, frame = cap.read()
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5) as hands:
            frame=cv2.flip(frame, 1)
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cv2.imshow("Capturing", frame)
        result = []
        landmarks = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    landmarks.append((landmark.x, landmark.y))
            for i in range(0, 21):
                for j in range(1,21):
                    if j > i:
                        result.append(math.sqrt( (landmarks[i][0] - landmarks[j][0])**2 + (landmarks[i][1] - landmarks[j][1])**2 ))
            result = np.array([result])

            predictions = model.predict(result)
            score = tf.nn.softmax(predictions[0])
            p = guesses[np.argmax(score)]
            path2 = cv2.imread(str(path2) + "{}_test.jpg".format(p))
            cv2.imshow("Prediction", path2)
        else:
            path2 = cv2.imread(str(path2) + "nothing_test.jpg")
            cv2.imshow("Prediction", path2)
        key=cv2.waitKey(1)
        if key == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()

predict_video("test2.mp4")
