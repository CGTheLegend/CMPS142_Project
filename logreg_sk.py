import random as rd
import numpy as np
import pandas as pd
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score

dataset = pd.read_csv('movie_metadata.csv')

director_FB_mean = dataset['director_facebook_likes'].mean()
cast_FB_mean = dataset['cast_total_facebook_likes'].mean()
imdb_mean = dataset['imdb_score'].mean()
movie_FB_mean = dataset['movie_facebook_likes'].mean()
num_voted_mean = dataset['num_voted_users'].mean()
# dataset[nan_set] = 0.0
dataset['success'] = ((dataset['gross'] > dataset['budget']) & 
	(((dataset['director_facebook_likes'] > director_FB_mean) | 
		(dataset['cast_total_facebook_likes'] > cast_FB_mean)) & 
	((dataset['imdb_score'] > imdb_mean) | (dataset['num_voted_users'] > num_voted_mean))) |
	(dataset['movie_facebook_likes'] > movie_FB_mean)).astype(int)

y, X = dmatrices("success ~ gross + budget + director_facebook_likes + \
	num_voted_users + cast_total_facebook_likes + budget + imdb_score + \
	movie_facebook_likes", dataset, return_type="dataframe")
y = np.ravel(y)

model = LogisticRegression()
model = model.fit(X, y)

print(model.score(X, y))
print(y.mean())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
modelV = LogisticRegression()
modelV.fit(X_train, y_train)
print y_train.mean()

predicted = modelV.predict(X_test)
print predicted

probs = modelV.predict_proba(X_test)
print probs

print metrics.accuracy_score(y_test, predicted)
