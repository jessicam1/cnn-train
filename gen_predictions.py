#!/usr/bin/env python


import sys
import argparse
import tensorflow as tf
from fast5fetch.fast5data import get_all_fast5s
from fast5fetch.fast5data import readid_x_generator_many_wrapper


def parse_args(args):
    parser = argparse.ArgumentParser(
        description="get neural network predictions for a dataset")
    parser.add_argument("-m", "--model",
                        help="neural network model to make predictions")
    parser.add_argument("-l", "--library", type=str,
                        help="path to guppy library to get predictions for")
    parser.add_argument("-w", "--window", type=int,
                        help="window length to use when sampling raw signal")
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
        limit_gpu(gpus[0], args.gpulim)

    model = tf.keras.models.load_model(args.model)

    fast5s = get_all_fast5s([args.library])

    gen = readid_x_generator_many_wrapper(
            fast5s, args.window, CPUS)

    readids_and_predictions(gen, model, args.window, args.threshold)

    # Alternatively we can use a dataset. This is more complex therefore not
    # preferred.
    # ds = tf.data.Dataset.from_generator(
    #         readid_x_generator_many_wrapper,
    #         args=[fast5s, args.window, CPUS],
    #         output_signature=(
    #             tf.TensorSpec(shape=(), dtype=tf.string),
    #             tf.TensorSpec(shape=(args.window, 1), dtype=tf.float32)))
    #
    # readids_and_predictions_from_ds(ds, model, args.window)


def limit_gpu(gpu_id, gpu_mem_lim):
    try:
        tf.config.experimental.set_virtual_device_configuration(
                gpu_id,
                [tf.config.experimental.VirtualDeviceConfiguration(
                    memory_limit=gpu_mem_lim)])
    except RuntimeError as err:
        print(err)


def readids_and_predictions(gen, model, window, threshold):
    for read_id, x in gen:
        # Reshape because the model expects a 3D input. The 1st dimension is
        # supposed to be the batch. However it doesn't have to match the
        # training batch size.
        x = x.reshape(1, window, 1)
        pred = model.predict(x)
        prediction = (pred >= threshold).astype("int32")
        print("\t".join([read_id, str(pred[0][0]), str(prediction[0][0])]))


# def readids_and_predictions_from_ds(ds, model, window):
#     for read_id, x in ds:
#         # Reshape because the model expects a 3D input. The 1st dimension is
#         # supposed to be the batch. However it doesn't have to match the
#         # training batch size.
#         x = tf.reshape(
#             x, [1, window, 1], name=None
#         )
#         pred = model.predict(x)
#         print("\t".join([read_id.numpy().decode("utf-8"), str(pred[0][0])]))


if __name__ == "__main__":
    main()
