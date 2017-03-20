import os
import tensorflow as tf
import numpy as np
# import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

dataset = np.genfromtxt('movie_metadata.csv', delimiter=",", names=('color', 'director', 'reviews', 'duration', 'director_likes', 'actor3_likes', 'actor2', 'actor1_likes', 'gross', 'genres', 'actor1', 'tittle', 'users', 'total_likes', 'actor3', 'posts', 'plot', 'imdb', 'num_review', 'language', 'country', 'rating', 'budget', 'year', 'actor2_likes', 'score', 'aspect_ratio', 'movie_likes'))
print(dataset)

# file_q = tf.train.string_input_producer(["movie_metadata.csv"])

# reader = tf.TextLineReader()
# key, value = reader.read(file_q)

# df = pd.read_csv("movie_metadata.csv")


# record_defaults = [[""],[""],[0],[0],[0],[0],[""],[0],[0],[""],[""],[""],[0],[0],[""],[0],[""],[""],[0],[""],[""],[""],[0],[0],[0],[0],[0],[0]]
# color, director, reviews, duration, director_likes, actor3_likes, actor2, actor1_likes, gross, genres, actor1, tittle, users, total_likes, actor3, posts, plot, imdb, num_review, language, country, rating, budget, year, actor2_likes, score, aspect_ratio, movie_likes = tf.decode_csv(value, record_defaults=record_defaults)
# features = tf.stack([color, director, reviews, duration, director_likes, actor3_likes, actor2, actor1_likes, gross, genres, actor1, tittle, users, total_likes, actor3, posts, plot, imdb, num_review, language, country, rating, budget, year, actor2_likes, score, aspect_ratio, movie_likes])
#
# with tf.Session() as sess:
#     tf.global_variables_initializer().run()
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord=coord)
#     while True:
#         try:
#             color_data, features_name = sess.run([color, features])
#             print(color_data, featuresy_name)
#         except tf.errors.OutOfRangeError:
#             break

# record_defaults = [[0] for row in range(5045)]
# dataset = tf.decode_csv(value, record_defaults=record_defaults)
# features = tf.stack([dataset[:]])


# Default values, in case of empty columns. Also specifies the type of the
# decoded result.
# record_defaults = [[1], [1], [1], [1], [1]]
# col1, col2, col3, col4, col5 = tf.decode_csv(
#     value, record_defaults=record_defaults)
# features = tf.stack([col1, col2, col3, col4])
#
# with tf.Session() as sess:
#   # Start populating the filename queue.
#   coord = tf.train.Coordinator()
#   threads = tf.train.start_queue_runners(coord=coord)
#
#   for i in range(1200):
#     # Retrieve a single instance:
#     example, label = sess.run([features, col5])
#
#   coord.request_stop()
#   coord.join(threads)
