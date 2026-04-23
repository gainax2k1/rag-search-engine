import argparse

from lib.inverted_index import InvertedIndex
from lib.search_utils import BM25_B, BM25_K1


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    
    _ = subparsers.add_parser("build", help="Build the search index")

    tf_parser = subparsers.add_parser("tf", help="Get term frequency for a document and term")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to get frequency for")

    idf_parser = subparsers.add_parser("idf", help="Get inverse document frequency for a term") 
    idf_parser.add_argument("term", type=str, help="Term to get frequency for")

    tfidf_parser = subparsers.add_parser("tfidf", help="Get TF-IDF score for a document and term")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term to get frequency for")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get frequency for")

    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a document and term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get frequency for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="BM25 k1 parameter (default: 1.5)")


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

        case "idf":
            print("Getting inverse document frequency...")
            idf_command(args.term) 

        case "tfidf":
            print("Getting TF-IDF score...")
            tfidf_command(args.doc_id, args.term)   

        case "bm25idf":
            print("Getting BM25 IDF score...")
            bm25_idf_command(args.term) 

        case "bm25tf":
            print("Getting BM25 TF score...")
            bm25_tf_command(args.doc_id, args.term, args.k1)
    
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

def idf_command(term):
    idx = InvertedIndex()
    idx.load()
    idf = idx.get_idf(term)
    print(f"Inverse document frequency of '{term}': {idf:.2f}")

def tfidf_command(doc_id, term):
    idx = InvertedIndex()
    idx.load()
    tf = idx.get_tf(doc_id, term)
    idf = idx.get_idf(term)
    tfidf = tf * idf
    print(f"TF-IDF score of '{term}' in document ID {doc_id}: {tfidf:.2f}")
    
def bm25_idf_command(term):
    idx = InvertedIndex()
    idx.load()
    idf = idx.get_bm25_idf(term)
    print(f"BM25 IDF score of '{term}': {idf:.2f}")    

def bm25_tf_command(doc_id, term, k1):
    idx = InvertedIndex()
    idx.load()
    bm25_tf = idx.get_bm25_tf(doc_id, term, k1)
    print(f"BM25 TF score of '{term}' in document ID {doc_id}: {bm25_tf:.2f}")

if __name__ == "__main__":
    main()