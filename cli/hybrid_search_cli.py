import argparse

from lib.search_utils import DEFAULT_ALPHA, DEFAULT_WSEARCH_LIMIT, load_movies
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

    args = parser.parse_args()

    match args.command:
        case "normalize":
            print("Normalizing scores...")
            normalize_command(args.scores)

        case "weighted-search":
            print("Weighted score search...")
            weighted_search_command(args.query, alpha = args.alpha, limit=args.limit)

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

    # still need to format and print results
    for i, result in enumerate(results):
        print(f"{i+1}. {result["title"]}")
        print(f"Hybrid Score: {result["hybrid"]:.4f}")
        print(f"BM25: {result["bm25"]:.4f}, Semantic: {result["sem"]:.4f}")
        print(f"{result["doc"][:100]}\n")



                                        




if __name__ == "__main__":
    main()