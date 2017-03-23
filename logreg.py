import os
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.contrib import learn

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

COLUMNS = ["color", "director_name", "num_critic_for_reviews", "duration", "director_facebook_likes", "actor_3_facebook_likes", 
    "actor_2_name", "actor_1_facebook_likes", "gross", "genres", "actor_1_name", "movie_title", "num_voted_users",
    "cast_total_facebook_likes", "actor_3_name", "facenumber_in_poster", "plot_keywords", "movie_imdb_link", "num_user_for_reviews",
    "language", "country", "content_rating", "budget", "title_year", "actor_2_facebook_likes", "imdb_score", "aspect_ratio",
    "movie_facebook_likes"]

dataset = np.array(pd.read_csv('movie_metadata.csv', names=COLUMNS, skipinitialspace=True))
data_train = np.array(pd.read_csv('movie_metadata.csv', names=COLUMNS, skipinitialspace=True))
data_test = np.array(pd.read_csv('movie_metadata.csv', names=COLUMNS, skipinitialspace=True))

prof_train = np.greater_equal(data_train[:, [8]], 2 * data_train[:, [22]]).astype(int)
prof_test = np.greater_equal(data_test[:, [8]], 2 * data_test[:, [22]]).astype(int)
# dataset[PROFIT_COLUMN] = np.greater_equal(data_train["gross"], 2 * data_train["budget"]).astype(int)
print(prof_train)
np.append(data_train, prof_train)
np.append(data_test, prof_test)
print(data_train.shape)
print(data_train)


director_facebook_likes = dataset["director_facebook_likes"]
gross = dataset["gross"]
num_voted_users = dataset["num_voted_users"]
cast_total_facebook_likes = dataset["cast_total_facebook_likes"]
num_user_for_reviews = dataset["num_user_for_reviews"]
budget = dataset["budget"]
imdb_scores = dataset["imdb_score"]
movie_facebook_likes = dataset["movie_facebook_likes"]

plot_keywords = dataset["plot_keywords"]
content_rating = dataset["content_rating"]
genres = dataset["genres"]
profitability = dataset["profitability"]

director_FB_likes = tf.contrib.layers.real_valued_column("director_facebook_likes")
bo_gross = tf.contrib.layers.real_valued_column("gross")
voted_users = tf.contrib.layers.real_valued_column("num_voted_users")
cast_total_FB_likes = tf.contrib.layers.real_valued_column("cast_total_facebook_likes")
film_budget = tf.contrib.layers.real_valued_column("budget")
imdb_ratings = tf.contrib.layers.real_valued_column("imdb_scores")
movie_FB_likes = tf.contrib.layers.real_valued_column("movie_facebook_likes")

keywords = tf.contrib.layers.sparse_column_with_hash_bucket("plot_keywords", hash_bucket_size = 100)
rating = tf.contrib.layers.sparse_column_with_keys("content_rating", keys=["G", "PG", "PG-13", "R", "NC-17"])
film_genres = tf.contrib.layers.sparse_column_with_hash_bucket("genres", hash_bucket_size = 100)

profit = tf.contrib.layers.sparse_column_with_keys("profitability", keys=[0, 1])

features = [director_FB_likes, bo_gross, film_genres, voted_users, 
	cast_total_FB_likes, keywords, rating, film_budget, imdb_ratings, movie_FB_likes]

model = tf.contrib.learn.LinearClassifier(feature_columns=features)
input_fn = tf.contrib.learn.io.numpy_input_fn({"director_facebook_likes":director_FB_likes, "gross":bo_gross, "genres":film_genres,
	"num_voted_users":voted_users, "cast_total_facebook_likes":cast_total_FB_likes, "plot_keywords":keywords,
	"content_rating":rating, "budget":film_budget, "imdb_score":imdb_ratings, "movie_facebook_likes":movie_FB_likes}, 
	data_train["profitability"], batch_size=100)
testing_fn = tf.contrib.learn.io.numpy_input_fn({"director_facebook_likes":director_FB_likes, "gross":bo_gross, "genres":film_genres,
	"num_voted_users":voted_users, "cast_total_facebook_likes":cast_total_FB_likes, "plot_keywords":keywords,
	"content_rating":rating, "budget":film_budget, "imdb_score":imdb_ratings, "movie_facebook_likes":movie_FB_likes}, 
	data_test["profitability"], batch_size=100)

print(model.fit(input_fn=input_fn, steps=200))
print("\n")
print(estimator.evaluate(input_fn=testing_fn))
print("\n")



