import argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

parser = argparse.ArgumentParser(
        description="creates convolutional neural network model")
parser.add_argument("-m", "--model", type=str, help="savepath of nn model")
parser.add_argument("-s", "--seqlength", type=int,
        help="sequence length to use as input")
args = parser.parse_args()

model = tf.keras.Sequential()
model.add(layers.BatchNormalization(input_shape=(args.seqlength,1),
    name="batchnorm"))
model.add(layers.Conv1D(64, kernel_size=30, strides=5, name="conv1"))
model.add(layers.MaxPool1D(pool_size=15, strides=3, name="mp1"))
model.add(layers.Conv1D(128, kernel_size=50, strides=12, name="conv2"))
model.add(layers.MaxPool1D(pool_size=5, strides=2, name="mp2"))
model.add(layers.Flatten())
model.add(layers.Dense(units=200, activation="relu", name="fc2"))
model.add(layers.Dense(units=10, activation="relu", name="fc3"))
model.add(layers.Dense(units=1, activation="sigmoid", name="sigmoid"))

model.compile(
        optimizer = 'adam',
        loss = 'binary_crossentropy',
        metrics = ['accuracy'],
        )

model.save(args.model)

model.summary()

