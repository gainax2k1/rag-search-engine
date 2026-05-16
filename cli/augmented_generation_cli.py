import argparse
from lib.search_utils import load_movies
from lib.hybrid_search import HybridSearch
from lib.query_enhancement import rag_evaluate, summarize_results, generate_citations, answer_question

def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser("rag", help="Perform RAG (search + generate answer)")
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser("summarize", help="Summarize search results")
    summarize_parser.add_argument("query", type=str, help="Search query for summarization")
    
    citations_parser = subparsers.add_parser("citations", help="Generate citations for search results")
    citations_parser.add_argument("query", type=str, help="Search query for citation generation")
    citations_parser.add_argument("--limit", type=int, nargs='?', default=5, help="Maximum number of documents to use for citation generation, default: {default})".format(default=5))    

    question_parser = subparsers.add_parser("question", help="Answer a question based on search results")
    question_parser.add_argument("question", type=str, help="Question to answer")
    question_parser.add_argument("--limit", type=int, nargs='?', default=5, help="Maximum number of documents to use for answering the question, default: {default})".format(default=5))    

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            print(f"Performing RAG for query: '{query}'")
            rag_command(query)

        case "summarize":
            query = args.query
            print(f"Summarizing results for query: '{query}'")
            summarize_command(query)

        case "citations":
            query = args.query
            limit = args.limit
            print(f"Generating citations for query: '{query}' with limit: {limit}")
            citations_command(query, limit)

        case "question":
            question = args.question
            limit = args.limit
            print(f"Answering question: '{question}' with limit: {limit}")
            question_command(question, limit)

        case _:
            parser.print_help()


def rag_command(query):
    documents = load_movies()
    hyb_search = HybridSearch(documents)
    results = hyb_search.rrf_search(query, k=60, limit=5)

    print("Search Results:")
    for result in results:
        print(f" - {result['title']}")
    
    rag_response = rag_evaluate(query, results)
    print("\nRAG Response:")
    print(rag_response)

def summarize_command(query):
    documents = load_movies()
    hyb_search = HybridSearch(documents)
    results = hyb_search.rrf_search(query, k=60, limit=5)

    print("Search Results:")
    for result in results:
        print(f" - {result['title']}")
    
    summary = summarize_results(query, results)
    print("\nLLM Summary:")
    print(summary)

def citations_command(query, limit):
    documents = load_movies()
    hyb_search = HybridSearch(documents)
    results = hyb_search.rrf_search(query, k=60, limit=limit)

    print("Search Results:")
    for result in results:
        print(f" - {result['title']}")
    
    citations = generate_citations(query, results)
    print("\nLLM Answer:")
    print(citations)

def question_command(question, limit):
    print(f"Generating answer for question: '{question}' with limit: {limit}")
    documents = load_movies()
    hyb_search = HybridSearch(documents)
    results = hyb_search.rrf_search(question, k=60, limit=limit)

    print("Search Results:")
    for result in results:
        print(f" - {result['title']}")
    
    answer = answer_question(question, results)
    print("\nAnswer:")
    print(answer)


if __name__ == "__main__":
    main()
