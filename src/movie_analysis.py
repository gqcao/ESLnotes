#!/bin/python

import numpy as np
import pandas as pd
from imdb import IMDb
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
sns.set_theme()
from sklearn.cluster import KMeans
from sklearn.datasets import load_svmlight_file
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE 
from common import loadstr, save2svm, get_sentence_model, normalize_feature
from pdb import set_trace

ia = IMDb()

class MovieAnalyzer():
    def __init__(self, movie_names, movie_info, feature_filename, release_years=None):
        self._movie_names       = movie_names
        self._movie_info        = movie_info
        self._feature_filename  = feature_filename 
        if release_years is None:
            self._release_years = [None] * len(movie_names)
        else:
            self._release_years = release_years  

    def _get_person_name(self, persons):
        if persons is None:
            return None
        else:
            names = ""
            for person in persons:
                names = names + person["name"] + ";"
            return names

    def _find_right_movie(self, movies, orig_movie_name, release_year=None):
        if len(movies) == 0:
            return None
        if release_year:
            for movie in movies:
                if movie.get("year") == release_year:
                    return movie
            print("None found..")
            return None
        else:
            print("Original name: " + orig_movie_name)
            for idx, movie in enumerate(movies):
                print("Movie number: ", idx)
                print(movie.get("title"))
                print(movie.get("year"))
            chose_idx = int(input("Choosing movie idx: "))
            return movies[chose_idx]

    def get_movie_info(self):
        movie_names = self._movie_names 
        movie_info_df = pd.DataFrame(columns=["IMDb_ID", "Title", "Directors", "Cast", "Year", "Rating", "Genre", "Top 250 Rank", "Runtimes"])
        plots = []
        for idx, (movie_name, release_year) in enumerate(zip(movie_names, self._release_years)):
            print("progress idx: " + str(idx))
            movie = self._find_right_movie(ia.search_movie(movie_name), movie_name, release_year)
            if movie is None:
                movie_info_df = movie_info_df.append({}, ignore_index=True)
                continue
            movie_imdb_info = ia.get_movie(movie.movieID)
            movie_row = {}
            movie_row["IMDb_ID"] = movie_imdb_info.get("imdbID")
            movie_row["Title"] = movie_imdb_info.get("title")
            movie_row["Directors"] = self._get_person_name(movie_imdb_info.get("directors"))
            movie_row["Cast"] = self._get_person_name(movie_imdb_info.get("cast"))
            movie_row["Year"] = movie_imdb_info.get("year")
            movie_row["Rating"] = movie_imdb_info.get("rating")
            if movie_imdb_info.get("genre"):
                movie_row["Genre"] = ";".join(movie_imdb_info.get("genre"))
            else:
                movie_row["Genre"] = None 
            movie_row["Top 250 Rank"] = movie_imdb_info.get("top 250 rank")
            plots.append(movie_imdb_info.get("plot outline"))
            if movie_imdb_info.get("runtimes"):
                movie_row["Runtimes"] = movie_imdb_info.get("runtimes")[0]
            else:
                movie_row["Runtimes"] = None
            movie_info_df = movie_info_df.append(movie_row, ignore_index=True)
        return movie_info_df, plots

    def _label_barplots(self, ax):
        for p in ax.patches:
            ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2,
                p.get_height()*1.005),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')

    def _show_year_distr(self, years):
        bins = list(range(1950, 2022))[0::10]
        bins.insert(0, min(years) - 1)
        binned_data = pd.cut(years, bins=bins, include_lowest=True)
        fig, ax = plt.subplots()
        ax = binned_data.value_counts(sort=False).plot.bar(rot=0, color="b", figsize=(10,7))
        ax.set_xticklabels(["Before 1950", "1950-1960", "1960-1970", "1970-1980", "1980-1990", "1990-2000", "2000-2010", "After 2010"])
        ax.set_xlabel("Years")
        ax.set_ylabel("Number of movies")
        self._label_barplots(ax)   # put the values on top of each bar..
        plt.title("Years of Movie Release")
        plt.savefig("plots/movie_year.jpg", dpi=150)
        #plt.show()
    
    def _show_rating_distr(self, ratings):
        bins = list(np.arange(6, 9.1, .5))
        bins.insert(0, 2)  # Add a very low 2 points in the front of the bins..
        binned_data = pd.cut(ratings, bins=bins, include_lowest=True)
        fig, ax = plt.subplots()
        ax = binned_data.value_counts(sort=False).plot.bar(rot=0, color="r", figsize=(9,7))
        ax.set_xticklabels(["<6", "6.0-6.5", "6.5-7.0", "7.0-7.5", "7.5-8.0", "8.0-8.5", "8.5-9.0"])
        ax.set_xlabel("Ratings")
        ax.set_ylabel("Number of movies")
        self._label_barplots(ax)   # put the values on top of each bar..
        plt.title("IMDb Ratings")
        plt.savefig("plots/movie_rating.jpg", dpi=150)
        #plt.show()
   
    def _print_good_movies(self, movie_info_df):
        good_movies = movie_info_df[movie_info_df["Rating"] > 8]["Title"].tolist()
        print("\n")
        print("Good movies:")
        print(good_movies)
        good_recent_movies = movie_info_df[(movie_info_df["Rating"] > 8) & (movie_info_df["Year"] > 1990)]["Title"].tolist()
        print("\n")
        print("Good recent movies:")
        print(good_recent_movies)
        print("\n")
        outstanding_movies = movie_info_df[movie_info_df["Rating"] > 8.5]["Title"].tolist()
        print("\n")
        print("Outstanding movies:")
        print(outstanding_movies)

    def _process_genres(self, genres):
        genres_flattened_list = []
        for genres in genres.tolist():
            genres_flattened_list.extend(genres.split(';'))
        return pd.Series(genres_flattened_list)

    def _show_genres(self, genres):
        genres_df = self._process_genres(genres)
        #labels, values = Counter(genres_list)
        fig, ax = plt.subplots()
        genres_hist = genres_df.value_counts(sort=True)
        genre_names = genres_hist.index.tolist()
        genres_hist.plot.bar(rot=0, color="g", figsize=(10,12))
        ax.set_xticklabels(genre_names, rotation=45)
        ax.set_xlabel("Genres")
        ax.set_ylabel("Number of movies")
        self._label_barplots(ax)   # put the values on top of each bar..
        plt.title("Movie Genre")
        plt.savefig("plots/genres.jpg", dpi=150)
        #plt.show()

    def _runtime_distr(self, movie_info_df):
        movies_80s = movie_info_df[(movie_info_df["Year"] > 1980) & (movie_info_df["Year"] < 1990)]
        movies_90s = movie_info_df[(movie_info_df["Year"] > 1990) & (movie_info_df["Year"] < 2000)]
        movies_00s = movie_info_df[(movie_info_df["Year"] > 2000) & (movie_info_df["Year"] < 2010)]
        cols = [movies_80s.Runtimes, movies_90s.Runtimes, movies_00s.Runtimes, movie_info_df.Runtimes.dropna()]
        fig, ax = plt.subplots(figsize=(6,5))
        c = "red"
        plt.boxplot(cols, positions=[1,2,3,4], notch=True, patch_artist=True,
                    boxprops=dict(facecolor=c, color=c),
                    capprops=dict(color=c),
                    whiskerprops=dict(color=c),
                    flierprops=dict(color=c, markeredgecolor=c),
                    medianprops=dict(color=c),
                    )
        ax.set_xticklabels(["1980s", "1990s", "2000s", "All"])
        ax.set_xlabel("Years")
        ax.set_ylabel("Runtimes")
        plt.title("Distribution of Runtimes by Years")
        plt.savefig("plots/runtime.jpg", dpi=150)
        #plt.show()

    def _visualize_plot_features(self, K=5):
        # Cluster the plot features into groups 
        features, labels        = load_svmlight_file(self._feature_filename)
        features                = normalize_feature(features.toarray())  # Normalize the feature 
        kmeans_model            = KMeans(K, random_state=0).fit(features)
        pred_labels             = kmeans_model.predict(features)
        transformed_features    = kmeans_model.transform(features)
        pca_model               = PCA(n_components=2)
        transformed_features_2d = pca_model.fit(transformed_features).transform(transformed_features)

        # Print out the grouped features
        movie_info_df = pd.read_csv(self._movie_info)
        genres = movie_info_df["Genre"]
        groups = {}
        for idx, name in enumerate(movie_names):
            entry = name + ": " + genres[idx] 
            if not groups.get(pred_labels[idx]):
                groups[pred_labels[idx]] = [entry]
            else:
                groups[pred_labels[idx]].append(entry)
        for idx in range(K):
            print("Group", idx)
            print(groups[idx])
            print("\n")

        # Visualize the grouped features
        plt.figure()
        colors = ["navy", "turquoise", "darkorange", "darkred", "chartreuse", "darkcyan", "darkviolet", "hotpink"]
        lw = 2
        for k in range(K):
            plt.scatter(
                transformed_features_2d[pred_labels==k, 0], transformed_features_2d[pred_labels==k, 1], color=colors[k], alpha=0.8, lw=lw
            )
        plt.legend(loc="best", shadow=False, scatterpoints=1)
        plt.title("2D Visualization of \"ESLNotes\" plots")
        plt.savefig("../plots/eslnotes_clusters.jpg", dpi=150)
        #plt.show()

    def analyze_movies(self):
        movie_info_df = pd.read_csv(self._movie_info)
        self._show_year_distr(movie_info_df["Year"])
        self._show_rating_distr(movie_info_df["Rating"])
        self._print_good_movies(movie_info_df)
        self._show_genres(movie_info_df["Genre"])
        self._runtime_distr(movie_info_df)
        self._visualize_plot_features(8)

class NetflixProcessor():
    def __init__(self, data_path):
        self._data_path = data_path

    def extract_movie_titles(self):
        data = pd.read_csv(self._data_path)
        movie_titles_df = data[(data["type"] == "Movie") & (data["country"] == "United States") & (data["release_year"] > 2015)]
        movie_titles_df.to_csv("../data/netflix_movies.csv", index=False)

    def find_highly_rated_movies(self, rating_thresh=6.5):
        movie_info                  = pd.read_csv("../data/netflix_movie_info.csv")
        highly_rated_movies         = movie_info[movie_info["Rating"] > rating_thresh]
        features, labels            = load_svmlight_file("../data/netflix_feature.txt")
        highly_rated_movie_features = features[movie_info["Rating"] > rating_thresh, :]
        return highly_rated_movies, highly_rated_movie_features

def analyze_eslnotes():
    # Analyze movies from eslnotes
    movie_list          = "../data/eslnotes_movie_list.txt"
    movie_names         = loadstr(movie_list)
    movie_info          = "../data/eslnotes_movie_info.csv"
    feature_filename    = "../data/eslnotes_feature.txt"
    eslnotes_analyzer = MovieAnalyzer(movie_names, movie_info, feature_filename)
    #movie_info_df,  plots = eslnotes_analyzer.get_movie_info()
    #movie_info_df.to_csv("../data/eslnotes_movie_info.csv", index=False)
    #writestr("../data/elnotes_plots.txt", plots)
    eslnotes_analyzer.analyze_movies()

def process_netflix_file():
    data_path = "../data/netflix_titles.csv"
    processor = NetflixProcessor(data_path)
    #processor.extract_movie_titles()
    highly_rated_movies, highly_rated_movie_features = processor.find_highly_rated_movies(7.5)
    highly_rated_movies["IMDb_ID"] = highly_rated_movies["IMDb_ID"].astype(int)
    highly_rated_movies["Year"] = highly_rated_movies["Year"].astype(int)
    highly_rated_movies.to_csv("../data/highly_rated_netflix_movie_info.csv", index=False)
    save2svm("../data/highly_rated_netflix_feature.txt", highly_rated_movie_features.toarray(), highly_rated_movies["Title"].tolist())

def analyze_netflix():
    # Analyze movies from eslnotes
    movie_list          = "../data/netflix_movies.csv"
    movie_names         = pd.read_csv(movie_list)["title"]
    movie_info          = "../data/netflix_movies.csv"
    feature_filename    = "../data/netflix_feature.txt"
    release_years       = pd.read_csv(movie_list)["release_year"]
    netflix_analyzer    = MovieAnalyzer(movie_names, movie_info, feature_filename, release_years)
    movie_info_df, plots = netflix_analyzer.get_movie_info()
    movie_info_df.to_csv("../data/netflix_movie_info.csv", index=False)

if __name__=="__main__":
    #analyze_eslnotes()
    #process_netflix_file()
    analyze_netflix()
