import argparse

from lib.search_utils import DEFAULT_ALPHA, DEFAULT_WSEARCH_LIMIT, K_WEIGHT, load_movies
from lib.hybrid_search import normalize_score, HybridSearch

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
            rrf_search_command(args.query, k=args.k, limit=args.limit)

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

def rrf_search_command(query, k, limit):
    documents = load_movies()
    hyb_search = HybridSearch(documents)
    results= hyb_search.rrf_search(query, k, limit)

    for i, result in enumerate(results):
        print(f"{i+1}. {result["title"]}")
        print(f"RRF Score: {result["rrf"]:.4f}")
        print(f"BM25 Rank: {result["bm25_rank"]}, Semantic Rank: {result["sem_rank"]}")
        print(f"{result["doc"][:100]}\n")

if __name__ == "__main__":
    main()