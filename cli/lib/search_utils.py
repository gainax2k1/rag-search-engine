import json, os, string

from nltk.stem import PorterStemmer


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

MOVIE_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")

BM25_K1 = 1.5 
BM25_B = 0.75



def load_movies():
    with open(MOVIE_DATA_PATH, "r") as fp:
        data = json.load(fp)
    return data["movies"]

def load_stopwords():
    with open(STOPWORDS_PATH, "r") as stop_fp:
        return set(stop_fp.read().splitlines())
    

def tokenize_text(text: str) -> list[str]:      
    stemmer = PorterStemmer() 
    stopwords = load_stopwords()
    tokens_list = preprocess_text(text).split()
    tokens_list = [stemmer.stem(token) for token in tokens_list if token not in stopwords]             
    return tokens_list

        
def preprocess_text(text: str) -> str:
    # remove punctuation, convert to lowercase, etc.
    text = text.strip()
    translator = str.maketrans("", "", string.punctuation)
    return text.translate(translator).lower()