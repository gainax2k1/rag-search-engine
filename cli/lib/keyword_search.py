from operator import add, index
import pickle, os, json, string

from nltk.stem import PorterStemmer
from .search_utils import load_movies, load_stopwords

stemmer = PorterStemmer()


INDEX_PATH = "cache/index.pkl"
DOCMAP_PATH = "cache/docmap.pkl"
CACHE_PATH = os.path.dirname(INDEX_PATH)
MAX_RETURNS = 5



class InvertedIndex:
    def __init__(self):
    # map string to set of integers (doc ids)
        self.index: dict[str, set[int]] = {}
    # map doc id to document text
        self.docmap: dict[int, dict] = {} 

    def __add_document(self, doc_id, text):

        #Tokenize the input text, then add each token to the index with the document ID.

        tokenized_text = tokenize_text(text)
        for word in tokenized_text:
            if word not in self.index:
                self.index[word] = set()
            self.index[word].add(doc_id)

    def search(self, query):
        words = tokenize_text(query)
        
        """
        
        if not words:
            return set()
            
        result = self.index.get(words[0], set())

        for word in words[1:]:
            # if any token matches
            result = result.union(self.index.get(word, set()))
            # if all tokens must match
            # re#sult = result.intersection(self.index.get(word, set()))
        """

        results = []
        seen = set()
        for word in words:
            for doc_id in sorted(self.index.get(word, [])):
                if doc_id not in seen:
                    seen.add(doc_id)
                    results.append(doc_id)
                    if len(results) == MAX_RETURNS:
                        return results
        return results

    

    def get_documents(self, term):
        tok_term = tokenize_text(term)
        if not tok_term:
            return []
        
        return self.index.get(tok_term[0], sorted(set()))
    
    def build(self):
        movies = load_movies()
        for m in movies:
            doc_id = m["id"]
            text = f"{m['title']} {m['description']}" 
            self.__add_document(doc_id, text)
            self.docmap[doc_id] = m
        
    def save(self): 
        os.makedirs(CACHE_PATH, exist_ok=True)
        with open(INDEX_PATH, 'wb') as f:
            pickle.dump(self.index, f)      
        with open(DOCMAP_PATH, 'wb') as f:
            pickle.dump(self.docmap, f)  

    def load(self):

        if not os.path.isfile(INDEX_PATH) or not os.path.isfile(DOCMAP_PATH):
            raise FileNotFoundError("Index or document map file not found")

        with open(INDEX_PATH, 'rb') as f:
            self.index = pickle.load(f)
        with open(DOCMAP_PATH, 'rb') as f:
            self.docmap = pickle.load(f)     


def tokenize_text(text: str) -> list[str]:       
    stopwords = load_stopwords()
    tokens_list = preprocess_text(text).split()
    tokens_list = [stemmer.stem(token) for token in tokens_list if token not in stopwords]             
    return tokens_list

        
def preprocess_text(text: str) -> str:
    # remove punctuation, convert to lowercase, etc.
    text = text.strip()
    translator = str.maketrans("", "", string.punctuation)
    return text.translate(translator).lower()
