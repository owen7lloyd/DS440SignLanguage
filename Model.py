import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd

# Creates model file for use in Main.py

def generate_model():
  data_dir = "landmarks.csv"

  hands = pd.read_csv("landmarks.csv")
  hand_features = hands.copy()
  hand_labels = hand_features.pop('actual')
  hand_features = np.array(hand_features).astype('float32')
  normalize = keras.layers.Normalization()
  normalize.adapt(hand_features)

  hand_model = tf.keras.Sequential([
        keras.layers.Dense(500),
        keras.layers.Dense(250),
        keras.layers.Dense(100),
        keras.layers.Dense(19, activation='softmax') # Must equal number of items being classified
        ])
  hand_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), optimizer = 'adam',  metrics=['accuracy'])
  hand_model.fit(x=hand_features, y=hand_labels, epochs=25)
  hand_model.save('Models\\model2.h5')

generate_model()