import os, json

from lib.semantic_search import SemanticSearch
from lib.search_utils import semantic_chunk, cosine_similarity, CACHE_DIR, SCORE_PRECISION

import numpy as np

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None # the vectors
        self.chunk_metadata = None # the index to connect to documents from chunks
        self.chunk_embeddings_path = os.path.join(CACHE_DIR, "chunk_embeddings.npy")
        self.chunk_metadata_path = os.path.join(CACHE_DIR, "chunk_metadata.json")

    def build_chunk_embeddings(self, documents):
        self.documents = documents

        #empty list of strings to hold chunks
        chunks = []
        #empty list of dictionaries to hold metadata
        meta_data = []

        # Build the document map
        for movie_idx, doc in enumerate(documents):
            # if description text is empty, skip
            if len(doc['description']) == 0:
                continue

            #add to docmapa
            self.document_map[doc['id']] = doc
            
            # Use your semantic chunking function to split the description text into 4-sentence chunks with 1-sentence overlap.
            chunked = semantic_chunk(doc['description'], 4, 1 )
            
            for chunk_idx, chunk in enumerate(chunked):
                chunks.append(chunk)
                entry = {
                    "movie_idx": movie_idx,
                    "chunk_idx": chunk_idx,
                    "total_chunks": len(chunked)
                }
                meta_data.append(entry)
        
        self.chunk_metadata = meta_data
        self.chunk_embeddings = self.model.encode(chunks, show_progress_bar = True)
        np.save(self.chunk_embeddings_path, self.chunk_embeddings)

        with open(self.chunk_metadata_path, "w") as f:
           #dump auto iterates through everything
           json.dump({"chunks": self.chunk_metadata, "total_chunks": len(chunks)}, f, indent=2)

        return self.chunk_embeddings    

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for doc in self.documents:
            self.document_map[doc['id']] = doc

        if os.path.exists(self.chunk_embeddings_path) and os.path.exists(self.chunk_metadata_path):
            self.chunk_embeddings = np.load(self.chunk_embeddings_path)
            with open(self.chunk_metadata_path, "r") as f:
                data = json.load(f)
            self.chunk_metadata = data["chunks"]
            return self.chunk_embeddings
        
        return self.build_chunk_embeddings(documents)
    
    def search_chunks(self, query: str, limit: int = 10):
        # Generate an embedding of the query (using the method from the SemanticSearch class)
        query_embedding =  self.generate_embeddings(query)

        # Populate an empty list to store "chunk score" dictionaries
        chunk_scores = []

        # For each chunk embedding:
        # - Calculate cos similarity
        # - append dictionary to chunk_scores with:
        #   - chunk_idx: index of chunk in the doc
        #   - movie_idx: index of the doc in self.documents using self.chunk_metadata to map to this)
        #   - score: cos similariy score

        for emb_idx, chunk in enumerate(self.chunk_embeddings):
            # compare vector of query with vectors in self
            cos_sim = cosine_similarity(chunk, query_embedding)

            # pull metadata from this chunk to assign to chunk_scores
            meta = self.chunk_metadata[emb_idx]
            score_entry = {
                "chunk_idx": meta["chunk_idx"],
                "movie_idx": meta["movie_idx"],
                "score": cos_sim
            }
            chunk_scores.append(score_entry)

        # empty DICTONARY maping movie_idx to score
        movie_scores = {}

        for score_record in chunk_scores:
            movie_idx = score_record["movie_idx"]
            score = score_record["score"]

            #if movie_idx not in movie_scores yet, or new score is higher, update with new chunk score
            existing_score = movie_scores.get(movie_idx)
            if (existing_score is None) or (existing_score < score):
                movie_scores[movie_idx] = score

        # sort movie_scores by score in descending order:
        
        sorted_movie_scores = sorted(movie_scores.items(), key=lambda pair:pair[1], reverse=True)

        results = []

        # iterates through sorted until limit or size of sorted is hit
        for movie_idx, score in sorted_movie_scores[:limit]:
            doc = self.documents[movie_idx]
            result_entry = {
                "id": doc["id"],
                "title": doc["title"],
                "document": doc["description"][:100],
                "score": round(score, SCORE_PRECISION),
                "metadata": {}
            }
            results.append(result_entry)

        return results