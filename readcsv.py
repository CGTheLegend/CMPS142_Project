import os
import tensorflow as tf
import pandas as pd


def readcsv(filename):
    # read in CSV with pandas
    ds = pd.read_csv(filename, sep=',', header=None)
    # convert to tensor
    dataset = tf.contrib.learn.extract_pandas_matrix(ds)
    features = tf.convert_to_tensor(dataset[0,:], dtype=tf.string)
    gross = tf.convert_to_tensor(dataset[1:,8], dtype=tf.float32)
    budget = tf.convert_to_tensor(dataset[1:,22], dtype=tf.float32)
    targets = tf.Variable(tf.zeros(5044), dtype=tf.float32)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        # generate targets values (gross>budget)
        for i in range(5043):
            if sess.run(gross[i]) > sess.run(budget[i]):
                sess.run(targets[i].assign(1))

    return dataset, features, targets
