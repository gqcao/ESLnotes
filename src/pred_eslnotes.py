#!/bin/python
from common import loadstr, writestr, save2svm, get_sentence_model, normalize_feature
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.datasets import load_svmlight_file
import pandas as pd
from sklearn.ensemble import IsolationForest
import numpy as np
from pdb import set_trace

class FeatureGenerator():
    def __init__(self, movie_names):
        self._movie_names = movie_names

    def gen_sentence_feature(self, sentences):
        model = get_sentence_model()
        features = []
        for sentence in sentences:
            features.append(model.encode(sentence, convert_to_tensor=True))
        save2svm("../data/feature.txt", features, self._movie_names)

class MovieIdentifier(): 
    def __init__(self, trn_feature_filename, tst_feature_filename, tst_movie_filename):
        self._trn_feature_filename  = trn_feature_filename
        self._tst_feature_filename  = tst_feature_filename
        self._tst_movie_filename    = tst_movie_filename

    def find_similar_movies(self):
        """Build a one-class classifier to determine whether new movies belong to the same cluster.
        """
        trn_features, trn_labels    = load_svmlight_file(self._trn_feature_filename)
        trn_features                = normalize_feature(trn_features.toarray())  # Normalize the feature 
        tst_features, tst_labels    = load_svmlight_file(self._tst_feature_filename)
        tst_features                = normalize_feature(tst_features.toarray())  # Normalize the feature
        oneclass_model              = IsolationForest(n_estimators=5, random_state=0).fit(trn_features)
        tst_labels                  = oneclass_model.predict(tst_features)
        tst_movie_names             = pd.read_csv(self._tst_movie_filename)["title"]
        writestr("../data/recommended_movies.txt", tst_movie_names.iloc[tst_labels == 1].tolist())

def gen_ESLnotes_features():
    movie_list = "../data/eslnotes_movie_list.txt"
    movie_names = loadstr(movie_list)
    fg = FeatureGenerator(movie_names)
    sentences = loadstr("../data/eslnotes_plots.txt")
    fg.gen_sentence_feature(sentences)

def gen_netflix_features():
    df = pd.read_csv("../data/netflix_movies.csv")
    movie_names = df["title"].tolist()
    sentences = df["description"].tolist()
    fg = FeatureGenerator(movie_names)
    fg.gen_sentence_feature(sentences)

def recommend_new_eslnotes_movies():
    trn_feature_filename    = "../data/eslnotes_feature.txt"
    tst_feature_filename    = "../data/netflix_feature.txt"
    tst_movie_filename      = "../data/netflix_movies.csv"
    eslnotes_identifier     = MovieIdentifier(trn_feature_filename, tst_feature_filename, tst_movie_filename)
    eslnotes_identifier.find_similar_movies()

if __name__ == "__main__":
    #gen_ESLnotes_features()
    #gen_netflix_features()
    recommend_new_eslnotes_movies()
