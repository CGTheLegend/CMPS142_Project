import tensorflow as tf
import pandas as pd


def readcsv(filename):
    # read in CSV with pandas
    ds = pd.read_csv(filename, sep=',', header=None)
    # pull useful tensors
    dataset = tf.contrib.learn.extract_pandas_matrix(ds)
    features = tf.convert_to_tensor(dataset[0,:], dtype=tf.string)
    gross = tf.convert_to_tensor(dataset[1:,8], dtype=tf.float32)
    budget = tf.convert_to_tensor(dataset[1:,22], dtype=tf.float32)

    return dataset, features, gross, budget
