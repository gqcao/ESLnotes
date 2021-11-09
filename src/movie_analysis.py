#!/bin/python

import numpy as np
import pandas as pd
from imdb import IMDb
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
sns.set_theme()
from pdb import set_trace

ia = IMDb()

class MovieAnalyzer():
    def __init__(self, movie_list):
        self._movie_list = movie_list

    def _get_person_name(self, persons):
        if persons is None:
            return None
        else:
            names = ""
            for person in persons:
                names = names + person["name"] + ";"
            return names

    def _find_right_movie(self, movies, orig_movie_name):
        if movies[0]["title"] != orig_movie_name:
            print("Original name: " + orig_movie_name)
            for idx, movie in enumerate(movies):
                print("Movie number: ", idx)
                print(movie.get("title"))
                print(movie.get("year"))
            chose_idx = int(input("Choosing movie idx: "))
            return movies[chose_idx]
        return movies[0]

    def get_movie_info(self):
        movie_names = self._loadstr(self._movie_list)
        movie_info_df = pd.DataFrame(columns=["IMDb_ID", "Title", "Directors", "Cast", "Year", "Rating", "Genre", "Top 250 Rank", "Runtimes"])
        plots = []
        for idx, movie_name in enumerate(movie_names):
            print("progress idx: " + str(idx))
            movie = self._find_right_movie(ia.search_movie(movie_name), movie_name)
            movie_imdb_info = ia.get_movie(movie.movieID)
            movie_row = {}
            movie_row["IMDb_ID"] = movie_imdb_info.get("imdbID")
            movie_row["Title"] = movie_imdb_info.get("title")
            movie_row["Directors"] = self._get_person_name(movie_imdb_info.get("directors"))
            movie_row["Cast"] = self._get_person_name(movie_imdb_info.get("cast"))
            movie_row["Year"] = movie_imdb_info.get("year")
            movie_row["Rating"] = movie_imdb_info.get("rating")
            movie_row["Genre"] = ";".join(movie_imdb_info.get("genre"))
            movie_row["Top 250 Rank"] = movie_imdb_info.get("top 250 rank")
            plots.append(movie_imdb_info.get("plot outline"))
            if movie_imdb_info.get("runtimes"):
                movie_row["Runtimes"] = movie_imdb_info.get("runtimes")[0]
            else:
                movie_row["Runtimes"] = None
            movie_info_df = movie_info_df.append(movie_row, ignore_index=True)
        movie_info_df.to_csv("data/eslnotes_movie_info.csv", index=False)
        self._writestr("data/elnotes_plots.txt", plots)

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

    def analyze_movies(self, movie_info):
        movie_info_df = pd.read_csv(movie_info)
        self._show_year_distr(movie_info_df["Year"])
        self._show_rating_distr(movie_info_df["Rating"])
        #self._print_good_movies(movie_info_df)
        self._show_genres(movie_info_df["Genre"])
        self._runtime_distr(movie_info_df)

class NetflixProcessor():
    def __init__(self, data_path):
        self._data_path = data_path

    def extract_movie_titles(self):
        data = pd.read_csv(self._data_path)
        movie_titles_df = data[(data["type"] == "Movie") & (data["country"] == "United States")]
        movie_titles_df.to_csv("../data/netflix_movies.csv", index=False)

def analyze_eslnotes():
    # Analyze movies from eslnotes
    movie_list = "../data/eslnotes_movie_list.txt"
    movie_info = "../data/eslnotes_movie_info.csv"
    eslnotes_analyzer = MovieAnalyzer(movie_list)
    #eslnotes_analyzer.get_movie_info()
    eslnotes_analyzer.analyze_movies(movie_info)

def analyze_netflix():
    data_path = "data/netflix_titles.csv"
    processor = NetflixProcessor(data_path)
    processor.extract_movie_titles()

if __name__=="__main__":
    #analyze_eslnotes()
    analyze_netflix()
