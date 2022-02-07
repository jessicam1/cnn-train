#!/usr/bin/env python
import sys
import time
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from fast5fetch.fast5data import get_all_fast5s
from fast5fetch.fast5data import train_test_val_split
from fast5fetch.fast5data import readid_x_generator_many
from fast5fetch.fast5data import readid_x_generator_many_wrapper

def parse_args(args):
    parser = argparse.ArgumentParser(description="get neural network predictions for a dataset")
    parser.add_argument("-m", "--model",
            help="neural network model to make predictions")
    parser.add_argument("-l", "--library", type=str,
            help="path to guppy library to get predictions for")
    parser.add_argument("-w", "--window", type=int,
            help="length of window to use when sampling raw signal")
    parser.add_argument("-b", "--batchsize", default=32, type=int,
            help="predictions batchsize; ensure same as batchsize model was trained")
    parser.add_argument("-t", "--threshold", default=0.5, type=float,
            help="sigmoid threshold value")
    parser.add_argument("-g", "--gpulim", default=4096, type=int,
            help="gpu memory limit in mb")
    parser.add_argument("-v", "--verbose", action="store_true",
            help="verbose model")
    return parser.parse_args(args)

def main():
    args = parse_args(sys.argv[1:])
    CPUS = 5
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        limit_gpu(gpus[0], 4096)

    model = tf.keras.models.load_model(args.model)

    gen_ds = build_ds_from_gen(args.library, args.window, args.batchsize, CPUS)
    readids_and_predictions(gen_ds, model, args.batchsize, args.window)

def limit_gpu(gpu_id, gpu_mem_lim):
    try:
        tf.config.experimental.set_virtual_device_configuration(
                gpu_id,
                [tf.config.experimental.VirtualDeviceConfiguration(
                    memory_limit = gpu_mem_lim)])
    except RuntimeError as err:
        print(err)

def build_ds_from_gen(library, window, batchsize, CPUS):
    dirs = []
    dirs += [library]
    fast5s = get_all_fast5s(dirs)
    gen_ds = tf.data.Dataset.from_generator(readid_x_generator_many_wrapper,
            args=[fast5s, window, True, CPUS],
            output_signature=(tf.TensorSpec(shape=(),
                dtype=tf.string),
                tf.TensorSpec(shape=(window,1), dtype=tf.float32)))

    row = [0] * window
    row_read = ["foo"]
    dummy_rows = []
    dummy_reads = []
    for i in range(batchsize):
        dummy_rows = np.append(dummy_rows, row, axis=0)
        dummy_reads.append(row_read)
    dummy_rows_tf = tf.constant(dummy_rows, dtype=tf.float32,
            shape=[batchsize, window, 1])
    dummy_reads_tf = tf.constant(dummy_reads, dtype=tf.string,
            shape=[batchsize,])
    dummy_ds = tf.data.Dataset.from_tensor_slices((
        dummy_reads_tf, dummy_rows_tf))

    gen_ds = gen_ds.concatenate(dummy_ds)

    return gen_ds

def readids_and_predictions(gen_ds, model, batchsize, window):
    i = 0
    batch = np.empty([32, window, 1])
    readids = []
    for readid, x in gen_ds:
        if i < batchsize:
            batch[i,:] = x.numpy().reshape([1, window, 1])
            i += 1
            readids.append(readid)
        if i == batchsize:
            preds = model.predict(batch)
            for j in range(len(batch)-1):
                print("\t".join([readids[j].numpy().decode("utf-8"),
                    str(preds[j][0])]))
            batch = np.empty([32, window, 1])
            # batch = []
            readids = []
            i = i - batchsize

if __name__ == "__main__":
    main()
