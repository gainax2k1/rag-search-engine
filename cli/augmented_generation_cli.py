import argparse
from lib.search_utils import load_movies
from lib.hybrid_search import HybridSearch
from lib.query_enhancement import rag_evaluate

def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser("rag", help="Perform RAG (search + generate answer)")
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            print(f"Performing RAG for query: '{query}'")
            rag_command(query)

        case _:
            parser.print_help()


def rag_command(query):
    documents = load_movies()
    hyb_search = HybridSearch(documents)
    results = hyb_search.rrf_search(query, k=60, limit=5)

    print("Search Results:")
    for result in results:
        print(f"- {result['title']}")
    
    rag_response = rag_evaluate(query, results)
    print("\nRAG Response:")
    print(rag_response)



if __name__ == "__main__":
    main()
