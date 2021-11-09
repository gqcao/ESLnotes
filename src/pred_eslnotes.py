#!/bin/python
from common import loadstr, save2svm, get_sentence_model
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from pdb import set_trace

class PredictESLnotes:
    def compute_similarity(self, s1, s2):
        #Sentences are encoded by calling model.encode()
        model = get_sentence_model()
        #Compute embedding for both lists
        embeddings1 = model.encode(s1, convert_to_tensor=True)
        embeddings2 = model.encode(s2, convert_to_tensor=True)

        #Compute cosine-similarits
        cosine_scores = cosine_similarity(embeddings1, embeddings2)

        #Output the pairs with their score
        for i in range(len(s1)):
            print("{} \t\t {} \t\t Score: {:.4f}".format(s1[i], s2[i], cosine_scores2[i][i]))

class FeatureGenerator():
    def __init__(self, movie_names):
        self._movie_names = movie_names

    def gen_sentence_feature(self, sentences):
        model = get_sentence_model()
        features = []
        for sentence in sentences:
            features.append(model.encode(sentence, convert_to_tensor=True))
        save2svm("../data/feature.txt", features, self._movie_names)

def test():
    pe = PredictESLnotes()
    # Two lists of sentences
    s1 = ['The cat sits outside',
                 'A man is playing guitar',
                 'The new movie is awesome']

    s2 = ['The dog plays in the garden',
                  'A woman watches TV',
                  'The new movie is so great']

    pe.compute_similarity(s1, s2)

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

if __name__ == "__main__":
    #gen_ESLnotes_features()
    gen_netflix_features()
