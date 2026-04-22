import argparse
from marshal import load

from lib.keyword_search import InvertedIndex


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    build_parser = subparsers.add_parser("build", help="Build the search index")

    tf_parser = subparsers.add_parser("tf", help="Get term frequency for a document and term")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to get frequency for")

    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            query_name = args.query.lower()
            print("Searching for:", query_name)
            #print("Cleaned query:", clean_text(query_name))
            idx = InvertedIndex()
            idx.load()
            results = idx.search(query_name)
            show_results(results, idx)
          
        case "build":
            print("Building index...")
            #build_index_and_docmap()
            build_command()

        case "tf":
            print("Getting term frequency...")
            tf_command(args.doc_id, args.term)
            

        case _:
            parser.print_help()


def build_command():
    idx = InvertedIndex()
    idx.build()
    idx.save()
    print("Index built and saved successfully.")
    
    """
    # removed for CH2L2
    docs = idx.get_documents("merida")
    print(f"First document for token 'merida' = {docs}")
    """
def show_results(results, idx):
    if not results:
        print("No results found.")
    else:
        print("Search results:")
    
        for doc_id in results:
                print(f"Document ID: {doc_id}, Title: {idx.docmap[doc_id]['title']}") 
    

def tf_command(doc_id, term):
    idx = InvertedIndex()
    idx.load()
    tf = idx.get_tf(doc_id, term)
    print(f"Term frequency of '{term}' in document ID {doc_id}: {tf}")

if __name__ == "__main__":
    main()