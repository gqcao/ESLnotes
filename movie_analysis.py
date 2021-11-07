#!/bin/python

import numpy as np
import pandas as pd
from imdb import IMDb
from pdb import set_trace
import matplotlib.pyplot as plt

ia = IMDb()

class AnalyzeMovies():
    def __init__(self, movie_list):
        self._movie_list = movie_list

    def _loadstr(self, filename, converter=str):
        return [converter(c.strip()) for c in open(filename).readlines()]

    def _get_person_name(self, persons):
        if persons is None:
            return None
        else:
            names = ""
            for person in persons:
                names = names + person["name"] + ";"
            return names

    def get_movie_info(self):
        movie_list = loadstr(self.movie_list)
        movie_info_df = pd.DataFrame(columns=["IMDb_ID", "Title", "Directors", "Cast", "Year", "Rating", "Genre", "Top 250 Rank", "Runtimes"])
        for idx, movie_name in enumerate(movie_list):
            print(idx)
            movie = ia.search_movie(movie_name)[0]
            movie_imdb_info = ia.get_movie(movie.movieID)
            movie_row = {}
            movie_row["IMDb_ID"] = movie_imdb_info.get("imdbID")
            movie_row["Title"] = movie_imdb_info.get("title")
            movie_row["Directors"] = get_person_name(movie_imdb_info.get("directors"))
            movie_row["Cast"] = get_person_name(movie_imdb_info.get("cast"))
            movie_row["Year"] = movie_imdb_info.get("year")
            movie_row["Rating"] = movie_imdb_info.get("rating")
            movie_row["Genre"] = ";".join(movie_imdb_info.get("genre"))
            movie_row["Top 250 Rank"] = movie_imdb_info.get("top 250 rank")
            if movie_imdb_info.get("runtimes"):
                movie_row["Runtimes"] = movie_imdb_info.get("runtimes")[0]
            else:
                movie_row["Runtimes"] = None
            movie_info_df = movie_info_df.append(movie_row, ignore_index=True)
        movie_info_df.to_csv("eslnotes_movie_info.csv", index=False)

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
        ax = binned_data.value_counts(sort=False).plot.bar(rot=0, color="g", figsize=(9,7))
        ax.set_xticklabels(["Before 1950", "1950-1960", "1960-1970", "1970-1980", "1980-1990", "1990-2000", "2000-2010", "After 2010"])
        ax.set_xlabel("Years")
        ax.set_ylabel("Number of movies")
        self._label_barplots(ax)   # put the values on top of each bar..
        plt.title("Years of Movie Release")
        plt.savefig("plots/movie_year.jpg", dpi=150)
        #plt.show()

    def analyze_movies(self, movie_info):
        movie_info_df = pd.read_csv(movie_info)
        self._show_year_distr(movie_info_df["Year"])

if __name__=="__main__":
    movie_list = "data/eslnotes_movie_list.txt"
    movie_info = "data/eslnotes_movie_info.csv"
    eslnotes_analyzer = AnalyzeMovies(movie_list)
    #eslnotes_analyzer.get_movie_info(movie_list)
    eslnotes_analyzer.analyze_movies(movie_info)
