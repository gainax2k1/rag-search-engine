from operator import add, index
import pickle, os, string, math

from nltk.stem import PorterStemmer
from .search_utils import load_movies, load_stopwords

stemmer = PorterStemmer()


INDEX_PATH = "cache/index.pkl"
DOCMAP_PATH = "cache/docmap.pkl"
TERM_FREQ_PATH = "cache/term_frequencies.pkl"
CACHE_PATH = os.path.dirname(INDEX_PATH)
MAX_RETURNS = 5
BM25_K1 = 1.5 
BM25_B = 0.75



class InvertedIndex:
    def __init__(self):
    # map string to set of integers (doc ids)
        self.index: dict[str, set[int]] = {}
    # map doc id to document text
        self.docmap: dict[int, dict] = {} 
    # map doc id to counter object
        self.term_frequencies: dict[int, dict[str, int]] = {}

    def __add_document(self, doc_id, text):

        #Tokenize the input text, then add each token to the index with the document ID.
        """update the term frequencies for each token in the document.
          For each token, increment its count in the Counter for that document ID."""
        tokenized_text = tokenize_text(text)
        for word in tokenized_text:
            if word not in self.index:
                self.index[word] = set()
            self.index[word].add(doc_id)
            # Update term frequency for the document
            if doc_id not in self.term_frequencies:
                self.term_frequencies[doc_id] = {}
            self.term_frequencies[doc_id][word] = self.term_frequencies[doc_id].get(word, 0) + 1

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
        with open(TERM_FREQ_PATH, 'wb') as f:
            pickle.dump(self.term_frequencies, f)

    def load(self):
        if not os.path.isfile(INDEX_PATH) or not os.path.isfile(DOCMAP_PATH) or not os.path.isfile(TERM_FREQ_PATH):
            raise FileNotFoundError("Index, document map, or term frequency file not found")

        with open(INDEX_PATH, 'rb') as f:
            self.index = pickle.load(f)
        with open(DOCMAP_PATH, 'rb') as f:
            self.docmap = pickle.load(f)
        with open(TERM_FREQ_PATH, 'rb') as f:
            self.term_frequencies = pickle.load(f)     

    def get_tf(self, doc_id, term):
        token_term = tokenize_text(term)
        if not token_term:
            return 0
        if len(token_term) > 1:
            raise ValueError("Term should be a single token after tokenization")
        
        tf = self.term_frequencies.get(doc_id, {}).get(token_term[0], 0)
        return tf
    
    def get_idf(self, term: str):
        token_term = tokenize_text(term)
        if not token_term:
            return 0
        if len(token_term) > 1:
            raise ValueError("Term should be a single token after tokenization")
        
        doc_freq = len(self.index[token_term[0]])
        print(f"Document frequency for term '{term}' (tokenized as '{token_term[0]}'): {doc_freq}")
        if doc_freq == 0:
            return 0
        
        total_docs = len(self.docmap)
        print(f"Total number of documents: {total_docs}")

        idf = math.log((total_docs + 1) / (doc_freq +1))  # Adding 1 to avoid division by zero and log(0)
        print(f"Inverse document frequency for term '{term}' (tokenized as '{token_term[0]}'): {idf}")
        return idf
    
    def get_bm25_idf(self, term: str) -> float:
        token_term = tokenize_text(term)
        if not token_term:
            return 0
        if len(token_term) > 1:
            raise ValueError("Term should be a single token after tokenization")
        
        doc_freq = len(self.index[token_term[0]])
        total_docs = len(self.docmap)

        bm25_idf = math.log((total_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1)
        return bm25_idf
    
    def get_bm25_tf(self, doc_id, term, k1=BM25_K1) -> float:
        token_term = tokenize_text(term)
        if not token_term:
            return 0
        if len(token_term) > 1:
            raise ValueError("Term should be a single token after tokenization")
        
        tf = self.get_tf(doc_id, term)
        doc_length = sum(self.term_frequencies.get(doc_id, {}).values())
        avg_doc_length = sum(sum(tf_dict.values()) for tf_dict in self.term_frequencies.values()) / len(self.term_frequencies)
        #doc length norm below
        # bm25_tf = (tf * (k1 + 1)) / (tf + k1 * (1 - 0.75 + 0.75 * (doc_length / avg_doc_length)))
        bm25_tf = (tf * (k1 + 1)) / (tf + k1)        
        return bm25_tf



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

