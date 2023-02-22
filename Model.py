import matplotlib.pyplot as plt
import numpy as np
import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
import gc
import pandas as pd

# actual,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210


gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

data_dir = "landmarks.csv"

hands = pd.read_csv("landmarks.csv")
hand_features = hands.copy()
hand_labels = hand_features.pop('actual')
hand_features = np.array(hand_features).astype('float32')
print(hand_features)
normalize = layers.Normalization()
normalize.adapt(hand_features)

hand_model = tf.keras.Sequential([
      layers.Dense(210),
      layers.Dense(24, activation='softmax')
      ])
hand_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), optimizer = 'adam',  metrics=['accuracy'])

hand_model.fit(x=hand_features, y=hand_labels, epochs=20)

# for letter in alphabet:
#   for file in os.listdir("{}".format(letter)):
#     hand = pd.read_csv("{}\\{}".format(letter,file))
#     hand_features = hand.copy()
#     hand_labels = [hand_features.pop('x'),hand_features.pop('y'),hand_features.pop('z')]
#     print(letter, file)

hand_model.save('model.h6')
