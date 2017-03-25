import os
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split


os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
np.set_printoptions(threshold='nan')

class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns

    def fit(self,X,y=None):
        return self

    def transform(self,X):
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)

COLUMNS = ["color", "director_name", "num_critic_for_reviews", "duration", "director_facebook_likes", "actor_3_facebook_likes",
    "actor_2_name", "actor_1_facebook_likes", "gross", "genres", "actor_1_name", "movie_title", "num_voted_users",
    "cast_total_facebook_likes", "actor_3_name", "facenumber_in_poster", "plot_keywords", "movie_imdb_link", "num_user_for_reviews",
    "language", "country", "content_rating", "budget", "title_year", "actor_2_facebook_likes", "imdb_score", "aspect_ratio",
    "movie_facebook_likes"]

# read in dataset
dataset_raw = np.array(pd.read_csv('movie_metadata.csv', names=COLUMNS, skipinitialspace=True))
# determine if movie was profitable or not
gross = dataset_raw[1:, 8]
budget = dataset_raw[1:, 22]
targets = np.array
targets = np.greater(gross, budget)
_NaNs = pd.isnull(targets)
targets[_NaNs] = False
# remove unquantifiable colums (i.e. names, iMDb link)
dataset_ = np.array
dataset_ = dataset_raw[1:,[2,3,4,5,7,8,9,12,13,15,18,19,20,21,22,23,24,25,26,27]]
#                          0 1 2 3 4 5 6  7  8  9 10 11 12 13 14 15 16 17 18 19
_NaNs = pd.isnull(dataset_)
dataset_[_NaNs] = 0
# encode lables
dataset = dataset_
le_genre = LabelEncoder()
dataset[:, 6] = le_genre.fit_transform(dataset_[:,6])
le_language = LabelEncoder()
dataset[:, 11] = le_language.fit_transform(dataset_[:,11])
le_country = LabelEncoder()
dataset[:, 12] = le_country.fit_transform(dataset_[:,12])
le_content_rating = LabelEncoder()
dataset[:, 13] = le_content_rating.fit_transform(dataset_[:,13])
# run SVC
clf = SVC()
clf.fit(dataset, targets)
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma='auto', kernel='sigmoid',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

# generate test dataset
X_train, X_test, y_train, y_test = train_test_split(dataset, targets, test_size=0.3, random_state=0.5)

predictions = clf.predict(X_test)
error = np.equal(y_test, predictions)
error_percent = np.sum(error)/1513 * 100
print  error_percent, "%"
