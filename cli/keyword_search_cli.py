import argparse, json, string

from pydoc import text
from nltk.stem import PorterStemmer


stemmer = PorterStemmer()

def clean_text(text: str) -> str:
    # remove punctuation, convert to lowercase, etc.
    text = text.strip()
    translator = str.maketrans("", "", string.punctuation)
    return text.translate(translator).lower()

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()



    with open("data/movies.json", "r") as fp:
        data = json.load(fp)
    movies = data["movies"]
    with open("data/stopwords.txt", "r") as stop_fp:
        stopwords = set(stop_fp.read().splitlines())
    

    match args.command:
        case "search":
            query_name = args.query.lower()
            print("Searching for:", query_name)
            #print("Cleaned query:", clean_text(query_name))

            counter = 0
            for movie in movies:
                # print("Checking movie:", movie["title"], "Cleaned title:", clean_text(movie["title"]))
                movie_tokens = clean_text(movie["title"]).split(" ")
                query_tokens = clean_text(query_name).split(" ")
                  
                for q_token in query_tokens:
                    for m_token in movie_tokens:
                        if q_token in stopwords or m_token in stopwords:
                            continue
                        if stemmer.stem(q_token) == stemmer.stem(m_token):
                            counter += 1    
                            print("Found:", movie["title"])
                            if counter == 5:
                                break
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()

