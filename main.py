import os
import tensorflow as tf
import pandas as pd
import readcsv as read
import findtargets as target

# ignore warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# read in CSV, pulling dataset, features and target(not yet generated)
dataset, features, gross, budget = read.readcsv('movie_metadata.csv')
targets = tf.Variable(tf.zeros(5044), dtype=tf.float32)

# iniitialize variables and Session
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# generates our targets bassed on profitabiliy (gross>budget)
targets = target.findtargets(sess, gross, budget, targets)
