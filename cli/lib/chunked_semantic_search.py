import os, json

from lib.semantic_search import SemanticSearch
from lib.search_utils import semantic_chunk, CACHE_DIR

import numpy as np


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None
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

            # self.document_map[doc['id']] = doc
            #movie_strings.append(f"{doc['title']}: {doc['description']}")
        
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
        # resume here, Celeste