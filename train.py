#!/usr/bin/env python

"""
train a neural network model.
"""

import sys
import random
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
from fast5fetch.fast5data import train_test_val_split
from fast5fetch.fast5data import data_and_label_generation


def parse_args(args):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-m", "--model", required=True,
            help="neural network model directory")
    parser.add_argument("-p", "--posdirs", required=True, nargs='+',
            help="fast5 directory containing positive label samples")
    parser.add_argument("-n", "--negdirs", nargs='+',
            help="fast5 directory containing negative label samples")
    parser.add_argument("--trainreads", required=True, type=int,
            help="number of reads to use for training")
    parser.add_argument("--valreads", required=True, type=int,
            help="number of reads to use for validation")
    parser.add_argument("--testreads", required=True, type=int,
            help="number of reads to use for testing model after training")
    parser.add_argument("-w", "--window", required=True, type=int,
            help="size of window for signal sampling")
    parser.add_argument("-r", "--ratio", type=float, default=0.5,
            help="ratio of positive to negative reads during training")
    parser.add_argument("-t", "--threshold", type=float, default=0.5,
            help="sigmoid threshold value, default is 0.5")
    parser.add_argument("-b", "--batchsize", type=int, default=32,
            help="number of reads to use in a training batch; default is 32")
    parser.add_argument("-l", "--logs",
            help="tensorboard log directory")
    parser.add_argument("-g", "--gpulim", type=int, default=4096,
            help="GPU memory limit in mb, default is 4096")
    parser.add_argument("--store_csv", type=str,
            help="give directory to build datasets and store in csv")
    parser.add_argument("--from_csv", type=str,
            help="give directory to build datasets from csv for training")
    return parser.parse_args(args)


def main():
    args = parse_args(sys.argv[1:])

    epoch_steps = args.trainreads / args.batchsize
    val_steps = args.valreads / args.batchsize
    pos_ratio = args.ratio
    neg_ratio = (1 - pos_ratio)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        limit_gpu(gpus[0], args.gpulim)

    model = tf.keras.models.load_model(args.model)

    if args.store_csv != '' or args.from_csv == '':
        pos_train, pos_val, pos_test = train_test_val_data(args.posdirs, 1, args.window)
        neg_train, neg_val, neg_test = train_test_val_data(args.negdirs, 0, args.window)
        train_ds, val_ds, test_ds = sample_data(
                pos_train, neg_train, pos_val, neg_val, pos_test, neg_test,
                args.batchsize, args.trainreads, args.valreads, args.testreads,
                args.ratio)

    if args.store_csv != '':
        store_csv(train_ds, val_ds, test_ds, args.store_csv)
        sys.exit(0)

    # if args.from_csv != '':
        # train_ds, val_ds, test_ds = dataset_from_csv() #FIXME

    # Train the model
    train_model(train_ds, val_ds, model, epoch_steps, val_steps, args.logs)

    # test_preds = testing(test_dataset, model, args.threshold)
    # tf.data.experimental.save(train_dataset, "neuralnets/src/train_dataset.csv", compression="gzip")


def store_csv(train_ds, val_ds, test_ds, path):
    store_single_csv(train_ds, path + '/train.csv')
    # store_single_csv(val_ds, path + '/val.csv')
    # store_single_csv(test_ds, path + '/test.csv')


def store_single_csv(ds, csv_file):
    for x, y in ds:
        for i in range(len(x)): # each i is one read
            read = x.numpy()[i] 
            # grab one read from batch and turn each signal into np array
            raw_data = [str(num[0]) for num in read] 
            label = str(y.numpy()[i])
            # take label from one read, convert to numpy, convert to to string
            print("\t".join([*raw_data, label]))
            # join all signals and label from one read

def limit_gpu(gpu_id, gpu_mem_lim):
    try:
        tf.config.experimental.set_virtual_device_configuration(
                gpu_id,
                [tf.config.experimental.VirtualDeviceConfiguration(
                    memory_limit = gpu_mem_lim)])
    except RuntimeError as err:
        print(err)


def train_test_val_data(file_dirs, label, window):
    train_list, test_list, val_list = train_test_val_split(file_dirs)

    train_set =  tf.data.Dataset.from_generator(
            data_and_label_generation,
            args=[train_list, label, window],
            output_signature=(tf.TensorSpec(shape=(window,1), dtype=tf.float32),
                              tf.TensorSpec(shape=(), dtype=tf.int16)))

    val_set =  tf.data.Dataset.from_generator(
            data_and_label_generation,
            args=[val_list, label, window],
            output_signature=(tf.TensorSpec(shape=(window,1), dtype=tf.float32),
                              tf.TensorSpec(shape=(), dtype=tf.int16)))

    test_set =  tf.data.Dataset.from_generator(
            data_and_label_generation,
            args=[test_list, label, window],
            output_signature=(tf.TensorSpec(shape=(window,1), dtype=tf.float32),
                              tf.TensorSpec(shape=(), dtype=tf.int16)))

    return train_set, val_set, test_set


def sample_data(pos_train, neg_train, pos_val, neg_val, pos_test, neg_test,
                batch_size, num_train, num_val, num_test, ratio):

    train_dataset =  tf.data.experimental.sample_from_datasets(
            [pos_train, neg_train],
            weights=[ratio, (1 - ratio)]).take(num_train).batch(batch_size)#.repeat(50)

    val_dataset =  tf.data.experimental.sample_from_datasets(
            [pos_val, neg_val],
            weights=[ratio, (1 - ratio)]).take(num_val).batch(batch_size).repeat(50)

    test_dataset =  tf.data.experimental.sample_from_datasets(
            [pos_test, neg_test],
            weights=[ratio, (1 - ratio)]).take(num_test).batch(batch_size).repeat(50)

    return train_dataset, val_dataset, test_dataset


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
            validation_steps = val_steps,
            )

    return fit


def testing(test_dataset, model, threshold):
    loss, acc = model.evaluate(test_dataset)
    predictions= model.predict(test_dataset)
    preds = (predictions >= threshold.astype("int32"))
    # preds.tolist()
    return preds


if __name__ == "__main__":
    main()



# train_list, val_list = train_val_split(args.posdirs)
# gen = data_and_label_generation(val_list, 1)
# for i in gen:
#     print(i)

# def test_generator(g):
#     counter = 0
#     for i in g:
#         counter += 1
#         print(i)
#         if counter > 10:
#             break

# counter = 0
# for x,y in train_dataset:
#     counter += 1
#     tf.print(list(y))
#     # print("\t".join([*str(y)]))
#     #tf.print(y.numpy())
#     
#     # rawdata = [str(num for num in list(y))]
#     # print("\t".join([*rawdata]))
#     # returns <generator object <genexpr> at blah>
#     if counter == 33:
#         break
# counter = 0
# for i in range(test_preds):
#     counter += 1
#     print(i)
#     if counter > 10:
#         break
