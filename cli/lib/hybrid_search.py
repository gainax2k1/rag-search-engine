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
        bm_results= self._bm25_search(query, limit * 500)
        semantic_results =  self.semantic_search.search_chunks(query, limit*500)

        bm_scores = []
        sem_scores = []
        scores_dict = {}

        for doc_id, score in bm_results:
            bm_scores.append(score)
            doc = self.idx.docmap[doc_id]
            entry = {
                "bm25": 0,
                "sem": 0,
                "hybrid": 0,
                "title": doc["title"],
                "doc": doc["description"]
            }
            scores_dict[doc_id] = entry

        for result in semantic_results:
            sem_scores.append(result["score"])
            doc_id = result["id"]
            if doc_id not in scores_dict:
                entry = {
                    "bm25": 0,
                    "sem": 0,
                    "hybrid": 0,
                    "title": result["title"],
                    "doc": result["document"],
                }
                scores_dict[doc_id] = entry
                    
        norm_bm = normalize_score(bm_scores)
        norm_sem = normalize_score(sem_scores)

        # fill in values here?
        for (doc_id, _), norm in zip(bm_results, norm_bm):
            scores_dict[doc_id]["bm25"] = norm

        for result, norm in zip(semantic_results, norm_sem):
            scores_dict[result["id"]]["sem"] = norm

        for entry in scores_dict.values():
            entry["hybrid"] = hybrid_score(entry["bm25"], entry["sem"], alpha)

        sorted_scores_dict = sorted(scores_dict.values(), key=lambda score:score["hybrid"], reverse=True)

        return sorted_scores_dict[:limit]

    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")
    

def normalize_score(scores: list[float]) -> list[float]:
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