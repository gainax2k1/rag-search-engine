import argparse, time

from lib.search_utils import DEFAULT_ALPHA, DEFAULT_WSEARCH_LIMIT, K_WEIGHT, load_movies
from lib.hybrid_search import normalize_score, HybridSearch
from lib.query_enhancement import enhance_query, individual_rerank, batch_rerank, cross_encoder_rerank

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")


    normalize_parser = subparsers.add_parser("normalize", help="Normalize list of scores")
    normalize_parser.add_argument("scores", nargs="*", type=float, help="list of scores to normalize")

    weighted_search_parser = subparsers.add_parser("weighted-search", help="Weighted search combining keyword and semantic scores")
    weighted_search_parser.add_argument("query", type=str, help="Search query")
    weighted_search_parser.add_argument("--alpha", nargs="?", type=float, default=DEFAULT_ALPHA, help="Alpha value for weighted search")
    weighted_search_parser.add_argument("--limit", nargs="?", type=int, default=DEFAULT_WSEARCH_LIMIT, help="Limit results of weighted search")

    rrf_search_parser = subparsers.add_parser("rrf-search", help="RRF search combining keyword and semantic scores")
    rrf_search_parser.add_argument("query", type=str, help="Search query")
    rrf_search_parser.add_argument("--k", nargs="?", type=int, default=K_WEIGHT, help="K value for RRF search")
    rrf_search_parser.add_argument("--limit", nargs="?", type=int, default=DEFAULT_WSEARCH_LIMIT, help="Limit results of RRF search")
    rrf_search_parser.add_argument("--enhance",type=str, choices=["spell", "rewrite", "expand"],help="Query enhancement method", default=None)
    rrf_search_parser.add_argument("--rerank-method", type=str, choices=["individual", "batch", "cross_encoder", None], help="Method for reranking results, default is to use the combined RRF score", default=None)


    args = parser.parse_args()

    match args.command:
        case "normalize":
            print("Normalizing scores...")
            normalize_command(args.scores)

        case "weighted-search":
            print("Weighted score search...")
            weighted_search_command(args.query, alpha = args.alpha, limit=args.limit)

        case "rrf-search":
            print("RRF score search...")
            rrf_search_command(args.query, k=args.k, limit=args.limit, enhance=args.enhance, rerank_method=args.rerank_method)

        case _:
            parser.print_help()

def normalize_command(scores):
    norm_scores = normalize_score(scores)

    for score in norm_scores:
        print(f"* {score:.4f}")


def weighted_search_command(query, alpha, limit):
    documents = load_movies()
    hyb_search = HybridSearch(documents)
    results= hyb_search.weighted_search(query, alpha, limit)

    for i, result in enumerate(results):
        print(f"{i+1}. {result["title"]}")
        print(f"Hybrid Score: {result["hybrid"]:.4f}")
        print(f"BM25: {result["bm25"]:.4f}, Semantic: {result["sem"]:.4f}")
        print(f"{result["doc"][:100]}\n")

def rrf_search_command(query, k, limit, enhance, rerank_method):
    documents = load_movies()
    hyb_search = HybridSearch(documents)

    if rerank_method in ("individual", "batch", "cross_encoder"):
        orig_limit = limit
        limit = limit * 5

    if enhance is None:
        results= hyb_search.rrf_search(query, k, limit)
    elif enhance in ("spell", "rewrite"):
        enhanced_query = enhance_query(query, method=enhance)
        print(f"Enhanced query ({enhance}): '{query}' -> '{enhanced_query}'\n")
        results = hyb_search.rrf_search(enhanced_query, k, limit)
    elif enhance == "expand":
        enhanced_query = query + " " + enhance_query(query, method=enhance)
        print(f"Enhanced query ({enhance}): '{query}' -> '{enhanced_query}'\n")
        results = hyb_search.rrf_search(enhanced_query, k, limit)
    else:
        raise ValueError(f"Invalid enhancement method: {enhance}")  
    
    if rerank_method == "individual":
    # run results through a series of llm promps (1 per doc) asking the llm to provide a new score for each  document
        print(f"Re-ranking top {orig_limit} results using individual method...")
        print(f"Reciprocal Rank Fusion Results for '{query}' (k={K_WEIGHT}):")
        for result in results[:limit]: # only rerank the top "limit" results"
            # prompt llm with result["doc"] and result["title"] and ask for a relevance score from 1-10
            result["individual_rerank"] = individual_rerank(query, title = result["title"], doc = result["doc"])
            time.sleep(3) # add delay to avoid rate limits    
        # sort results by individual_rerank score instead of RRF score
        results = sorted(results, key=lambda x: x["individual_rerank"], reverse=True)[:limit//5] # return top "limit" results after reranking

    elif rerank_method == "batch":
        for i, r in enumerate(results):
            r["id"] = i  # add an "id" field to each result based on its index in the results list, since batch_rerank expects an "id" for each doc to identify them in the reranking process   

        ranked_ids = batch_rerank(query, results)[:orig_limit]

        rank_by_id = {id: i + 1 for i, id in enumerate(ranked_ids)}
        for result in results:
            if result["id"] in rank_by_id:
                result["batch_rerank"] = rank_by_id[result["id"]]

        results = sorted(results, key=lambda x: x.get("batch_rerank", float("inf")))[:orig_limit]
        print(f"Re-ranking top {orig_limit} results using batch method...")
        print(f"Reciprocal Rank Fusion Results for '{query}' (k={K_WEIGHT}):\n")
    
    if rerank_method == "cross_encoder":
        results = cross_encoder_rerank(query, results[:limit])[:orig_limit]
        print(f"Re-ranking top {orig_limit} results using cross_encoder method...")
        print(f"Reciprocal Rank Fusion Results for '{query}' (k={K_WEIGHT}):\n")    

    for i, result in enumerate(results):
        print(f"{i+1}. {result["title"]}")
        if rerank_method == "individual":
            print(f"   Re-rank Score: {result["individual_rerank"]:.3f}/10")
        if rerank_method == "batch":
            print(f"   Re-rank Rank: {i+1}")
        if rerank_method == "cross_encoder":
            print(f"   Cross Encoder Score: {result["cross_encoder_score"]:.3f}")   
      
        print(f"   RRF Score: {result["rrf"]:.3f}")
        print(f"   BM25 Rank: {result["bm25_rank"]}, Semantic Rank: {result["sem_rank"]}")
        print(f"   {result["doc"][:100]}\n")

if __name__ == "__main__":
    main()