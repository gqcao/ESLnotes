#!/bin/python

import numpy as np
import pandas as pd
from imdb import IMDb
from pdb import set_trace

ia = IMDb()

def loadstr(filename,converter=str):
    return [converter(c.strip()) for c in open(filename).readlines()]

def get_person_name(persons):
    if persons is None:
        return None
    else:
        names = ""
        for person in persons:
            names = names + person["name"] + ";"
        return names

def get_movie_info(movie_list):
    movie_list = loadstr(movie_list)
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

if __name__=="__main__":
    movie_list = "eslnotes_movie_list.txt"
    get_movie_info(movie_list)
