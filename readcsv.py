import os
import tensorflow as tf
import pandas as pd


def readcsv(filename):
    # Read in CSV with pandas
    ds = pd.read_csv(filename, sep=',', header=None)
    # convert to tensor
    dataset = tf.contrib.learn.extract_pandas_matrix(ds)
    features = tf.convert_to_tensor(dataset[0,:], dtype=tf.string)
    gross = tf.convert_to_tensor(dataset[1:,8], dtype=tf.float32)
    return dataset, features, gross
