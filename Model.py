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

# Generates Model.h5 for use in Main.py

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

hand_model.save('model.h5')
