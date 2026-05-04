import os

from .inverted_index import InvertedIndex
from .chunked_semantic_search import ChunkedSemanticSearch

class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit):
        bm_val = self._bm25_search(query, limit * 500)
        chunk_sem_val =  self.semantic_search.search_chunks(query, limit*500)

        norm_bm_val =


    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")
    

def normalize_score(scores[float])=> list[float]:
    if len(scores) == 0:
        return []
    
    min_val = min(scores)
    max_val = max(scores)
    
    norm_scores = []

    if min_val == max_val:
        for score in scores:
            norm_scores.append(1.0)
        return norm_scores
    
    for score in scores:
        normed = (score- min_val) / (max_val-min_val)
        norm_scores.append(normed)
    return norm_scores


def hybrid_score(bm25_score, semantic_score, alpha=0.5):
    return alpha * bm25_score + (1 - alpha) * semantic_score