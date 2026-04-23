import pickle, os, string, math

from .search_utils import load_movies, tokenize_text, BM25_K1, BM25_B, CACHE_DIR

MAX_RETURNS = 5

class InvertedIndex:
    def __init__(self):
    # map string to set of integers (doc ids)
        self.index: dict[str, set[int]] = {}
    # map doc id to document text
        self.docmap: dict[int, dict] = {} 
    # map doc id to counter object
        self.term_frequencies: dict[int, dict[str, int]] = {}
    # map doc id to document length
        self.doc_lengths: dict[int, int] = {}

        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.term_freq_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")

    def __add_document(self, doc_id, text):
        tokenized_text = tokenize_text(text)
        doc_length = 0

        for word in tokenized_text:
            doc_length += 1
            if word not in self.index:
                self.index[word] = set()
            self.index[word].add(doc_id)
            # Update term frequency for the document
            if doc_id not in self.term_frequencies:
                self.term_frequencies[doc_id] = {}
            self.term_frequencies[doc_id][word] = self.term_frequencies[doc_id].get(word, 0) + 1
        self.doc_lengths[doc_id] = doc_length

    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def search(self, query):
        words = tokenize_text(query)

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
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(self.index_path, 'wb') as f:
            pickle.dump(self.index, f)      
        with open(self.docmap_path, 'wb') as f:
            pickle.dump(self.docmap, f)  
        with open(self.term_freq_path, 'wb') as f:
            pickle.dump(self.term_frequencies, f)
        with open(self.doc_lengths_path, 'wb') as f:
            pickle.dump(self.doc_lengths, f)

    def load(self):
        if not os.path.isfile(self.index_path) or not os.path.isfile(self.docmap_path) or not os.path.isfile(self.term_freq_path):
            raise FileNotFoundError("Index, document map, or term frequency file not found")

        with open(self.index_path, 'rb') as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, 'rb') as f:
            self.docmap = pickle.load(f)
        with open(self.term_freq_path, 'rb') as f:
            self.term_frequencies = pickle.load(f)   
        with open(self.doc_lengths_path, 'rb') as f:
            self.doc_lengths = pickle.load(f)  

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
    
    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B) -> float:
        token_term = tokenize_text(term)
        if not token_term:
            return 0
        if len(token_term) > 1:
            raise ValueError("Term should be a single token after tokenization")
        
        tf = self.get_tf(doc_id, term)
        doc_length = sum(self.term_frequencies.get(doc_id, {}).values())
        avg_doc_length = self.__get_avg_doc_length()
        #doc length norm below
        bm25_tf = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_length / avg_doc_length)))
          
        return bm25_tf
    
    def bm25(self, doc_id, term, k1=BM25_K1, b=BM25_B) -> float:
        idf = self.get_bm25_idf(term)
        tf = self.get_bm25_tf(doc_id, term, k1, b)
        return idf * tf
    
    def bm25_search(self, query, limit=MAX_RETURNS, k1=BM25_K1, b=BM25_B):
        words = tokenize_text(query)
        scores = {} # map doc_id to BM25 score

        for doc in self.docmap:
            total_bm25_score = 0
            for word in words:
                total_bm25_score += self.bm25(doc, word, k1, b)
            scores[doc] = total_bm25_score

        
        ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]
        return ranked_docs

