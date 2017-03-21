import os
import tensorflow as tf
import pandas as pd
import readcsv as read

# ignore warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

dataset = read.readcsv('movie_metadata.csv')
print(dataset)
