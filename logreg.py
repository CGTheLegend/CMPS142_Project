from tensorflow.contrib import learn
import os
import tensorflow as tf
import numpy as np
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

data_train, data_test = [],[]

COLUMNS = ["color", "director_name", "num_critic_for_reviews", "duration", "director_facebook_likes", "actor_3_facebook_likes",
    "actor_2_name", "actor_1_facebook_likes", "gross", "genres", "actor_1_name", "movie_title", "num_voted_users",
    "cast_total_facebook_likes", "actor_3_name", "facenumber_in_poster", "plot_keywords", "movie_imdb_link", "num_user_for_reviews",
    "language", "country", "content_rating", "budget", "title_year", "actor_2_facebook_likes", "imdb_score", "aspect_ratio",
    "movie_facebook_likes", "profitability"]

data_train = np.array(pd.read_csv('movie_metadata.csv', names=COLUMNS, skipinitialspace=True))
data_test = np.array(pd.read_csv('movie_metadata.csv', names=COLUMNS, skipinitialspace=True))
nans_train = pd.isnull(data_train)
data_train[nans_train] = 0.0
nans_test = pd.isnull(data_test)
data_test[nans_test] = 0.0

color                     = data_train[:, [0]]
director_name             = data_train[:, [1]]
num_critic_for_reviews    = data_train[:, [2]]
duration                  = data_train[:, [3]]
director_facebook_likes   = data_train[:, [4]]
actor_3_facebook_likes    = data_train[:, [5]]
actor_2_name              = data_train[:, [6]]
actor_1_facebook_likes    = data_train[:, [7]]
gross                     = data_train[:, [8]]
actor_1_name              = data_train[:, [10]]
movie_title               = data_train[:, [11]]
num_voted_users           = data_train[:, [12]]
cast_total_facebook_likes = data_train[:, [13]]
actor_3_name              = data_train[:, [14]]
facenumber_in_poster      = data_train[:, [15]]
movie_imdb_link           = data_train[:, [17]]
num_user_for_reviews      = data_train[:, [18]]
language                  = data_train[:, [19]]
country                   = data_train[:, [20]]
budget                    = data_train[:, [22]]
title_year                = data_train[:, [23]]
actor_2_facebook_likes    = data_train[:, [24]]
imdb_score                = data_train[:, [25]]
aspect_ratio              = data_train[:, [26]]
movie_facebook_likes      = data_train[:, [27]]

plot_keywords = data_train[:, [16]]
content_rating = data_train[:, [21]]
genres = data_train[:, [9]]
profitability = data_train[:, [28]]

director_FB_likes = tf.contrib.layers.real_valued_column("director_facebook_likes")
bo_gross = tf.contrib.layers.real_valued_column("gross")
voted_users = tf.contrib.layers.real_valued_column("num_voted_users")
cast_total_FB_likes = tf.contrib.layers.real_valued_column("cast_total_facebook_likes")
film_budget = tf.contrib.layers.real_valued_column("budget")
imdb_ratings = tf.contrib.layers.real_valued_column("imdb_score")
movie_FB_likes = tf.contrib.layers.real_valued_column("movie_facebook_likes")

keywords = tf.contrib.layers.sparse_column_with_hash_bucket("plot_keywords", hash_bucket_size = 100)
rating = tf.contrib.layers.sparse_column_with_keys("content_rating", keys=["G", "PG", "PG-13", "R", "NC-17"])
film_genres = tf.contrib.layers.sparse_column_with_hash_bucket("genres", hash_bucket_size = 100)

profit = tf.contrib.layers.sparse_column_with_keys("profitability", keys=[0, 1])

features = [tf.contrib.layers.real_valued_column("director_facebook_likes", dimension=1),
	tf.contrib.layers.real_valued_column("gross", dimension=1),
	tf.contrib.layers.real_valued_column("num_voted_users", dimension=1),
	tf.contrib.layers.real_valued_column("cast_total_facebook_likes", dimension=1),
	tf.contrib.layers.real_valued_column("budget", dimension=1),
	tf.contrib.layers.real_valued_column("imdb_score", dimension=1),
	tf.contrib.layers.real_valued_column("movie_facebook_likes", dimension=1)]

print(data_train[4,:])
model = tf.contrib.learn.LinearClassifier(feature_columns=features)
input_fn = tf.contrib.learn.io.numpy_input_fn({"color":color, "director_name":director_name, "num_critic_for_reviews":num_critic_for_reviews,
	"duration":duration, "director_facebook_likes":director_facebook_likes, "actor_3_facebook_likes":actor_3_facebook_likes,
	"actor_2_name":actor_2_name, "actor_1_facebook_likes":actor_1_facebook_likes, "gross":gross, "genres":genres, "actor_1_name":actor_1_name,
	"movie_title":movie_title, "num_voted_users":num_voted_users, "cast_total_facebook_likes":cast_total_facebook_likes,
	"actor_3_name":actor_3_name, "facenumber_in_poster":facenumber_in_poster, "plot_keywords":plot_keywords, "movie_imdb_link":movie_imdb_link,
	"num_user_for_reviews":num_user_for_reviews, "language":language, "country":country, "content_rating": content_rating ,"budget":budget,
	"title_year":title_year, "actor_2_facebook_likes":actor_2_facebook_likes,"imdb_score":imdb_score, "aspect_ratio":aspect_ratio,
	"movie_facebook_likes":movie_facebook_likes}, profitability, batch_size=100, num_epochs=500)

sess = tf.Session()
print("OUTPUT: \n")
print(model.fit(input_fn=input_fn))
