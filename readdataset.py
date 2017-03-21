import os
import tensorflow as tf
import numpy as np
import pandas as pd

# ignore warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Read in CSV with pandas
ds = pd.read_csv('movie_metadata.csv', sep=',', header=None)
print(ds)

# convert to tensor
dataset = tf.contrib.learn.extract_pandas_matrix(ds)
sess = tf.Session()
print(dataset)
