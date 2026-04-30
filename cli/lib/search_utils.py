import json, os, string
import numpy as np
from nltk.stem import PorterStemmer


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")

MOVIE_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
MOVIE_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "movie_embeddings.npy")

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

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)