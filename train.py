#!/usr/bin/env python

"""
train a neural network model.
"""

import sys
import argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
from fast5fetch.fast5data import train_test_val_split
from fast5fetch.fast5data import xy_generator_many
from fast5fetch.fast5data import xy_generator_many_wrapper


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
    parser.add_argument("-c", "--cache", action="store_true",
            help="use caching to speed up training")
    return parser.parse_args(args)


def main():
    args = parse_args(sys.argv[1:])
    max_epochs = 80
    epoch_steps = args.trainreads / args.batchsize
    val_steps = args.valreads / args.batchsize
    pos_ratio = args.ratio
    neg_ratio = (1 - pos_ratio)
    CPUS = 5

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        limit_gpu(gpus[0], args.gpulim)

    modelpath = args.model
    model = tf.keras.models.load_model(modelpath)

    pos_train_list, pos_test_list, pos_val_list = train_test_val_split(args.posdirs)
    neg_train_list, neg_test_list, neg_val_list = train_test_val_split(args.negdirs)

    pos_train_ds = make_class_ds(
            pos_train_list, 1, args.window, args.ratio, args.trainreads, CPUS)
    pos_test_ds = make_class_ds(
            pos_test_list, 1, args.window, args.ratio, args.testreads, CPUS)
    pos_val_ds = make_class_ds(
            pos_val_list, 1, args.window, args.ratio, args.valreads, CPUS)
    neg_train_ds = make_class_ds(
            neg_train_list, 0, args.window, 1 - args.ratio, args.trainreads, CPUS)
    neg_test_ds = make_class_ds(
            neg_test_list, 0, args.window, 1 - args.ratio, args.testreads, CPUS)
    neg_val_ds = make_class_ds(
            neg_val_list, 0, args.window, 1 - args.ratio, args.valreads, CPUS)

    train_ds = build_full_ds(pos_train_ds, neg_train_ds, args.batchsize,
            args.ratio, max_epochs, args.cache)
    val_ds = build_full_ds(pos_val_ds, neg_val_ds, args.batchsize,
            args.ratio, max_epochs, args.cache)

    # Train the model
    train_model(train_ds, val_ds, model, epoch_steps, val_steps, modelpath, max_epochs, args.logs)


def limit_gpu(gpu_id, gpu_mem_lim):
    try:
        tf.config.experimental.set_virtual_device_configuration(
                gpu_id,
                [tf.config.experimental.VirtualDeviceConfiguration(
                    memory_limit = gpu_mem_lim)])
    except RuntimeError as err:
        print(err)

def make_class_ds(file_list, label, window, ratio, numreads, CPUS):
    num = int(numreads*ratio)
    class_ds =  tf.data.Dataset.from_generator(
            xy_generator_many_wrapper,
            args=[file_list, label, window, True, CPUS],
            output_signature=(tf.TensorSpec(shape=(window,1), dtype=tf.float32),
                              tf.TensorSpec(shape=(), dtype=tf.int16)))
    ds = class_ds.take(num)
    return ds 

def build_full_ds(pos_ds, neg_ds, batchsize, ratio, max_epochs, cache):
    ds =  tf.data.experimental.sample_from_datasets(
            [pos_ds, neg_ds], weights=[ratio, (1 - ratio)],
            stop_on_empty_dataset=True)
    # ds = ds.repeat(5)
    if cache:
        ds= ds.cache()
    ds = ds.batch(batchsize)
    ds = ds.repeat(max_epochs)
    return ds

def train_model(train_dataset, val_dataset, model, epoch_steps, val_steps, modelpath, max_epochs, logs):

    model.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001), # 0.0005),
            loss = 'binary_crossentropy',
            metrics = ["accuracy"],
            )

    callbacks = [
            # keras.callbacks.EarlyStopping(
            #     monitor="val_loss",
            #     min_delta=1e-2,
            #     patience=5,
            #     verbose=1
            #     ),
            keras.callbacks.ModelCheckpoint(
                filepath = modelpath,
                save_best_only=True,
                monitor="val_loss",
                verbose=1,
                ),
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
            epochs=max_epochs,
            steps_per_epoch = epoch_steps,
            callbacks=callbacks,
            validation_data=val_dataset,
            validation_steps = val_steps,
            )
    # model.save("neuralnets/mycnn/models/test_save/")

    return fit

if __name__ == "__main__":
    main()
