import os
import tensorflow as tf
import pandas as pd


def readcsv(filename):
    # Read in CSV with pandas
    ds = pd.read_csv(filename, sep=',', header=None)
    # convert to tensor
    dataset = tf.contrib.learn.extract_pandas_matrix(ds)
    return dataset
