import argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

parser = argparse.ArgumentParser(description="creates convolutional neural network model")
parser.add_argument("-m", "--model", type=str, help="savepath of nn model")
args = parser.parse_args()

seq_length = 4000 
model = tf.keras.Sequential()
model.add(layers.BatchNormalization(input_shape=(seq_length,1), name="batchnorm"))
# model.add(layers.Conv1D(128, kernel_size=125, strides=15, name="conv1")) # 128 400 30
# model.add(layers.MaxPool1D(pool_size=45, strides=10, name="mp1")) 

model.add(layers.Conv1D(64, kernel_size=80, strides=10, name="conv1")) # 128 400 30
model.add(layers.MaxPool1D(pool_size=25, strides=7, name="mp1")) # 35 7
model.add(layers.Conv1D(128, kernel_size=15, strides=3, name="conv2")) # 128, 7 , 3
model.add(layers.MaxPool1D(pool_size=5, strides=2, name="mp2")) # 4 2 # 5 2
# model.add(layers.Conv1D(192, kernel_size=5, strides=2, name="conv3")) # 32 2 2 # 128 5 2
# model.add(layers.MaxPool1D(pool_size=3, strides=2, name="mp3")) # 2 2 # 3 2
model.add(layers.Flatten())
#model.add(layers.Dense(units=400, activation="relu", name="fc1")) # 200
model.add(layers.Dense(units=200, activation="relu", name="fc2")) # 200
model.add(layers.Dense(units=10, activation="relu", name="fc3"))
model.add(layers.Dense(units=1, activation="sigmoid", name="sigmoid"))

model.compile(
        optimizer = 'adam',
        loss = 'binary_crossentropy',
        metrics = ['accuracy'],
        )

model.save(args.model)

model.summary()

