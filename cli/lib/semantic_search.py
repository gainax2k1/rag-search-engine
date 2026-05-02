from sentence_transformers import SentenceTransformer
from lib.search_utils import MOVIE_EMBEDDINGS_PATH, CACHE_DIR, load_movies, cosine_similarity
import os
import numpy as np

SEARCH_LIMIT = 5

class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        # Load the model (downloads automatically the first time)
        self.model = SentenceTransformer(model_name)

        # numpy 2d array of vectors
        """self.embeddings = np.array([
                [0.12, -0.34, 0.56, ...],   # vector for documents[0] = "The Great Bear"
                [0.78,  0.11, -0.22, ...],  # vector for documents[1] = "Spaceflight IC-1"
                [-0.05, 0.42, 0.31, ...],   # vector for documents[2] = "Adventureland"
                # ...
            ])"""
        self.embeddings = None
 
        # dictionary of movie data, with 'id', 'title', and 'description' keys, indexed by doc_id
        """for example:
            self.documents = [
                {"id": 0, "title": "The Great Bear",   "description": "A bear who..."},
                {"id": 1, "title": "Spaceflight IC-1", "description": "The opening..."},
                {"id": 2, "title": "Adventureland",    "description": "In 1987..."},
                # ...
            ]"""
        self.documents = None

        # map doc_id to full document
        """self.document_map = {
            0: {"id": 0, "title": "The Great Bear", ...},
            1: {"id": 1, "title": "Spaceflight IC-1", ...},
            # ...
        }"""
        self.document_map: dict[int, str] = {} # double check this type hint, should it be dict[int, str]?

        self.embeddings_path = MOVIE_EMBEDDINGS_PATH

    def encode(self, text):
        return self.model.encode(text)
    
    def generate_embeddings(self, text):
        if len(text.strip()) == 0:
            raise ValueError("Empty text cannot encode in generate_embeddings.")
        
        text_list = [text] # since the model expects a list of sentences, we wrap the input text in a list

        return self.model.encode(text_list)[0]
    
    

    def build_embeddings(self, documents): #builds and saves the embeddings for the given documents, and also stores the documents in the instance variable for later use in search results
        self.documents = documents

        movie_strings = [] #list to store the strings in titile: description format for each movie
        # Build the document map
        for doc in documents:
            self.document_map[doc['id']] = doc
            movie_strings.append(f"{doc['title']}: {doc['description']}")
        
        self.embeddings = self.model.encode(movie_strings, show_progress_bar = True)
        np.save(self.embeddings_path, self.embeddings)

        return self.embeddings

        
    def load_or_create_embeddings(self, documents):
        self.documents = documents # store the documents in the instance variable for later use in search results
        if os.path.exists(self.embeddings_path):
            print("Loading cached embeddings...")
            self.embeddings = np.load(self.embeddings_path)
        else:
            print("Generating new embeddings...")
            self.build_embeddings(documents)
        
        if len(documents) != len(self.embeddings):
            self.build_embeddings(documents)
        
        return self.embeddings
    
    def search(self, query, limit=SEARCH_LIMIT):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        
        query_embedding = self.generate_embeddings(query)

        # Calculate cosine similarity between the query embedding and each document embedding.
        scores = []
        
        for i in range(len(self.embeddings)):
            score = cosine_similarity(query_embedding, self.embeddings[i])
            scores.append((score, self.documents[i]))
        # 
        
        sorted_scores = sorted(scores, key=lambda x: x[0], reverse=True)
        sim_list = sorted_scores[:limit]        
        
        results = []

        """Return the top results (up to limit) as a list of dictionaries, each containing:

            score: The cosine similarity score
            title: The movie title
            description: The movie description
        """

        for score, doc in sim_list:

            results.append({
                "score": score,
                "title": doc["title"],
                "description": doc["description"]
            })
  
        return results
    
def verify_embeddings():
    sem_search = SemanticSearch()
    movie_list = load_movies()
    embeddings = sem_search.load_or_create_embeddings(movie_list)
    print(f"Number of docs:   {len(movie_list)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_query_text(query):
    sem_search = SemanticSearch()
    query_list = sem_search.generate_embeddings(query)

    print(f"Query: {query}")
    print(f"First 3 dimensions: {query_list[:3]}")
    print(f"Shape: {query_list.shape}")
    

