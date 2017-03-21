import os
import tensorflow as tf
import pandas as pd
import readcsv as read

# ignore warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# read in CSV, pulling dataset, features and target
dataset, features, targets = read.readcsv('movie_metadata.csv')

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
