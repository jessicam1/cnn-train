#!/usr/bin/env python
from Fast5Fetch.Fast5Data_dev import train_val_split
from Fast5Fetch.Fast5Data_dev import data_generation
# from Fast5Fetch.Fast5Data_dev import SampleGeneratorFromFiles
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
import argparse

parser = argparse.ArgumentParser(description="train a neural network model.")
parser.add_argument("-m", "--model", required=True, help="neural network model directory")
parser.add_argument("-p", "--posdirs", required=True, nargs='+', help="fast5 directory containing positive label samples")
parser.add_argument("-n", "--negdirs", nargs='+', help="fast5 directory containing negative label samples")
parser.add_argument("--trainreads", required=True, type=int, help="number of reads to use for training")
parser.add_argument("--valreads", required=True, type=int, help="number of reads to use for validation")
parser.add_argument("-r", "--ratio", type=float, default=0.5, help="ratio of positive to negative reads during training")
parser.add_argument("-t", "--threshold", type=float, default=0.5, help="sigmoid threshold value, default is 0.5")
parser.add_argument("-b", "--batchsize", type=int, default=32, help="number of reads to use in a training batch; default is 32") 
# parser.add_argument("--numreads", type=int, default=100000, help="number of reads to use for training + validation")
parser.add_argument("-l", "--logs", help="tensorboard log directory")
parser.add_argument("-g", "--gpulim", type=int, default=4096, help="GPU memory limit in mb, default is 4096")
# parser.add_argument("-p", "--pattern", action="store_true", help="insert pattern (fifty consecutive '1's) into positive samples for training purposes")
args = parser.parse_args()
# add option for window

epoch_steps = args.trainreads / args.batchsize
val_steps = args.valreads / args.batchsize
pos_ratio = args.ratio
neg_ratio = (1 - pos_ratio)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit = args.gpulim)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)


model = tf.keras.models.load_model(args.model)

def train_val_data(file_dirs, label):
    train_list, val_list = train_val_split(file_dirs) #, num_train, num_val)
    # train_gen = SampleGeneratorFromFiles(train_list, label)
    # val_gen = SampleGeneratorFromFiles(val_list, label)
    train_set =  tf.data.Dataset.from_generator(data_generation, args=[train_list, label], output_signature=(tf.TensorSpec(shape=(8000,1), dtype=tf.float32), tf.TensorSpec(shape=(), dtype=tf.int16))) 
    val_set =  tf.data.Dataset.from_generator(data_generation, args=[train_list, label], output_signature=(tf.TensorSpec(shape=(8000,1), dtype=tf.float32), tf.TensorSpec(shape=(), dtype=tf.int16))) 
    return train_set, val_set

def sample_data(pos_train, neg_train, pos_val, neg_val, batch_size, num_train, num_val, ratio):

    train_dataset =  tf.data.experimental.sample_from_datasets([pos_train, neg_train], weights=[ratio, (1 - ratio)]).take(num_train).batch(batch_size).repeat()
    val_dataset =  tf.data.experimental.sample_from_datasets([pos_val, neg_val], weights=[ratio, (1 - ratio)]).take(num_val).batch(batch_size).repeat()

    return train_dataset, val_dataset


def train_model(train_dataset, val_dataset, model, epoch_steps, val_steps, logs):

    model.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005), 
            loss = 'binary_crossentropy',
            metrics = ["accuracy"],
            )

    callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                min_delta=1e-2,
                patience=5,
                verbose=1
                ),
            # keras.callbacks.ModelCheckpoint(
            #     filepath = model,
            #     save_best_only=True,
            #     monitor="val_loss",
            #     verbose=1,
            #     ),
            # keras.callbacks.ReduceLROnPlateau(
            #     monitor="val_loss",
            #     factor=0.5,
            #     patience=3,
            #     ),
            # tf.keras.callbacks.TensorBoard(
            #     log_dir=logs,
            #     histogram_freq=1,
            #     ),
            ]

    fit = model.fit(
            train_dataset,
            epochs=50,
            steps_per_epoch = epoch_steps,
            callbacks=callbacks,
            validation_data=val_dataset,
            validation_steps = val_steps, # ? 
            )

    return fit

# train_list, val_list = train_val_split(args.posdirs)
# gen = data_generation(val_list, 1)
# for i in gen:
#     print(i)

# def test_generator(g):
#     counter = 0
#     for i in g:
#         counter += 1
#         print(i)
#         if counter > 10:
#             break

pos_train, pos_val = train_val_data(args.posdirs, 1)
neg_train, neg_val = train_val_data(args.negdirs, 0)
train_dataset, val_dataset = sample_data(pos_train, neg_train, pos_val, neg_val, args.batchsize, args.trainreads, args.valreads, args.ratio)
fit = train_model(train_dataset, val_dataset, model, epoch_steps, val_steps, args.logs)
