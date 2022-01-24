#!/usr/bin/env python

"""
train a neural network model.
"""

import sys
import time
import random
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
from fast5fetch.fast5data import train_test_val_split
from fast5fetch.fast5data import data_and_label_generation
from fast5fetch.fast5data import xy_generator_many


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
    parser.add_argument("-c", "--cache", action="store_true",
            help="use caching to speed up training")
    return parser.parse_args(args)


def main():
    args = parse_args(sys.argv[1:])
    max_epochs = 50
    epoch_steps = args.trainreads / args.batchsize
    val_steps = args.valreads / args.batchsize
    pos_ratio = args.ratio
    neg_ratio = (1 - pos_ratio)
    CPUS = 5

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        limit_gpu(gpus[0], args.gpulim)

    model = tf.keras.models.load_model(args.model)

    # if args.store_csv != '' or args.from_csv == '':
    pos_train, pos_val, pos_test = train_test_val_data(
                args.posdirs, 1, args.window, CPUS, args.cache)
    neg_train, neg_val, neg_test = train_test_val_data(
                args.negdirs, 0, args.window, CPUS, args.cache)
    train_ds = sample_data(
                pos_train, neg_train, args.batchsize, args.trainreads, args.ratio)
    val_ds = sample_data(
                pos_val, neg_val, args.batchsize, args.valreads, args.ratio)
    test_ds = sample_data(
                pos_test, neg_test, args.batchsize, args.testreads, args.ratio)

    # if args.store_csv != '':
    #     store_csv(train_ds, val_ds, test_ds, args.window, args.store_csv)
    #     sys.exit(0)
    #
    # if args.from_csv != '':
    #     train_ds, val_ds, test_ds = dataset_from_csv(
    #             args.from_csv, args.window, args.batchsize) #FIXME

    # Train the model
    train_model(train_ds, val_ds, model, epoch_steps, val_steps, args.logs,
            args.cache)

    # test_preds = testing(test_dataset, model, args.threshold)
    # tf.data.experimental.save(train_dataset, "neuralnets/src/train_dataset.csv", compression="gzip")


def store_csv(train_ds, val_ds, test_ds, window, path):
    store_single_csv(train_ds, window, path + '/train.csv')
    # store_single_csv(val_ds, window, path + '/val.csv')
    # store_single_csv(test_ds, window, path + '/test.csv')


def store_single_csv(ds, window, csv_file):
    t0 = time.time()
    with open(csv_file, "w") as csv:
        col_array = np.arange(0, window+1, 1)
        cols = [str(num) for num in col_array]
        csv.write("\t".join([*cols, "label"]) + "\n")
        for x, y in ds:
            for i in range(len(x)):#.numpy())): # each i is one read
                read = x[i].numpy() 
                # grab one read from batch and turn each signal into np array
                raw_data = [str(num[0]) for num in read] 
                label = str(y[i].numpy())
                # # take label from one read, convert to numpy then to str
                print("\t".join([*raw_data, label]))
                csv.write("\t".join([*raw_data, label]) + "\n")
                # join all signals and label from one read
    t1=time.time()
    print(t1 - t0, file=sys.stderr)

def dataset_from_csv(path, window, bs, ratio):
    train_ds = tf.data.experimental.make_csv_dataset(
            path/"train.csv", header=True, batch_size=bs, label_name="label")
    val_ds = tf.data.experimental.make_csv_dataset(
            path/"val.csv", header=True, batch_size=bs, label_name="label")
    test_ds = tf.data.experimental.make_csv_dataset(
            path/"test.csv", header=True, batch_size=bs, label_name="label")
    iterator = train_ds.as_numpy_iterator()
    print(next(iterator))
    # try num_parallel_reads arg
    # shuffle = numreads
    # ratio?
    return train_ds, val_ds, test_ds
    
def limit_gpu(gpu_id, gpu_mem_lim):
    try:
        tf.config.experimental.set_virtual_device_configuration(
                gpu_id,
                [tf.config.experimental.VirtualDeviceConfiguration(
                    memory_limit = gpu_mem_lim)])
    except RuntimeError as err:
        print(err)


def train_test_val_data(file_dirs, label, window, CPUS, cache):
    train_list, test_list, val_list = train_test_val_split(file_dirs)

    train_set =  tf.data.Dataset.from_generator(
            xy_generator_many,
            args=[train_list, label, window, True, CPUS],
            output_signature=(tf.TensorSpec(shape=(window,1), dtype=tf.float32),
                              tf.TensorSpec(shape=(), dtype=tf.int16)))

    val_set =  tf.data.Dataset.from_generator(
            xy_generator_many,
            args=[val_list, label, window, True, CPUS],
            output_signature=(tf.TensorSpec(shape=(window,1), dtype=tf.float32),
                              tf.TensorSpec(shape=(), dtype=tf.int16)))

    test_set =  tf.data.Dataset.from_generator(
            xy_generator_many,
            args=[test_list, label, window, True, CPUS],
            output_signature=(tf.TensorSpec(shape=(window,1), dtype=tf.float32),
                              tf.TensorSpec(shape=(), dtype=tf.int16)))

    if cache==False:
        train_set = train_set.repeat()
        val_set = val_set.repeat()
        test_set = test_Set.repeat()

    return train_set, val_set, test_set


def sample_data(pos_set, neg_set, batch_size, numreads, ratio):

    ds =  tf.data.experimental.sample_from_datasets(
            [pos_set, neg_set],
            weights=[ratio, (1 - ratio)]).take(numreads).batch(batch_size)#.repeat()

    return ds

def train_model(train_dataset, val_dataset, model, epoch_steps, val_steps, logs, cache):
    epochs = 60
    if cache:
        train_dataset = train_dataset.cache().repeat(epochs)

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
            epochs=epochs,
            steps_per_epoch = epoch_steps,
            # callbacks=callbacks,
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
