import semantic_search

import numpy as np


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents):
        # 
        self.load_or_create_chunk_embeddings(documents)
        #empty list of strings to hold chunks
        chunks = []
        #empty list of dictionaries to hold metadata
        meta_data = [{}]

        return
    

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        return