import json

#load_movies()
#load_stopwords()

MOVIE_DATA_PATH = "data/movies.json"
STOPWORDS_PATH = "data/stopwords.txt"



def load_movies():
    with open(MOVIE_DATA_PATH, "r") as fp:
        data = json.load(fp)
    return data["movies"]

def load_stopwords():
    with open(STOPWORDS_PATH, "r") as stop_fp:
        return set(stop_fp.read().splitlines())