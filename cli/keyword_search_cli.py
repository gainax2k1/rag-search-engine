import argparse

from lib.keyword_search import InvertedIndex


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    build_parser = subparsers.add_parser("build", help="Build the search index")

    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            query_name = args.query.lower()
            print("Searching for:", query_name)
            #print("Cleaned query:", clean_text(query_name))

          
        case "build":
            print("Building index...")
            #build_index_and_docmap()
            build_command()



        case _:
            parser.print_help()


def build_command():
    idx = InvertedIndex()
    idx.build()
    idx.save()
    print("Index built and saved successfully.")
    docs = idx.get_documents("merida")
    print(f"First document for token 'merida' = {docs}")
    # ... print the verification message



if __name__ == "__main__":
    main()