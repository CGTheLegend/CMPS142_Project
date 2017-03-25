import os
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder


os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns

    def fit(self,X,y=None):
        return self

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
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
# remove unquantifiable colums (i.e. names)
dataset_ = np.array
dataset_ = dataset_raw[1:,[2,3,4,5,7,8,12,13,15,18,22,23,24,25,26,27]]
_NaNs = pd.isnull(dataset_)
dataset_[_NaNs] = 0
# encode lables
dataset = MultiColumnLabelEncoder(columns = [9,19,20,2]).fit_transform(dataset_)

clf = SVC()
clf.fit(dataset, targets)

SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

# read in test dataset
test = dataset[4500:,:]

predictions = clf.predict(dataset)
error = np.equal(targets[4500:], predictions)
print(np.sum(error))
