import argparse, json
from lib.search_utils import load_movies
from lib.hybrid_search import HybridSearch

def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    # run evaluation logic here
    golden_ds = json.load(open("data/golden_dataset.json", "r"))

    hybrid_search = HybridSearch(load_movies())

    # for each query in the golden dataset, we would run the search and compare results to the golden dataset answers to calculate precision@k, recall@k, and F1@k
  
    print(f"Evaluating RRF Search with k={limit}")
    for entry in golden_ds["test_cases"]:
        query = entry["query"]
        relevant_docs = entry["relevant_docs"]
             
        # rrf's k set to 60 as per intructions.
        # limit is the "top k" (number of results)
        hybrid_search_results = hybrid_search.rrf_search(query, k=60, limit=limit)  

        retrieved_titles = set(result["title"] for result in hybrid_search_results)


        relevant_titles = set(relevant_docs)

        total_relevant = len(relevant_titles)   

        relevant_retrieved = retrieved_titles & relevant_titles
        precision = len(relevant_retrieved) / len(hybrid_search_results)
        recall = len(relevant_retrieved) / total_relevant if total_relevant > 0 else 0
        harmonic_mean = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0 # "F1" score
        
        print(f"\n- Query: {query}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Recall@{limit}: {recall:.4f}")
        print(f"  - F1 Score: {harmonic_mean:.4f}")
        print(f"  - Retrieved: {", ".join(retrieved_titles)}")
        print(f"  - Relevant: {", ".join(relevant_titles)}")   


if __name__ == "__main__":
    main()
"""
            rrf_results['doc_id']
            bm25_rank = entry["bm25_rank"]
            sem_rank = entry["sem_rank"]

            bm25_score = rrf_score(bm25_rank, k) if bm25_rank is not None else 0
            sem_score = rrf_score(sem_rank, k) if sem_rank is not None else 0
            entry["rrf"] = bm25_score + sem_score 
"""