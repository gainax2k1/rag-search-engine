#!/usr/bin/env python3

import argparse, re

from torch import embedding
from lib.semantic_search import SemanticSearch, verify_embeddings, embed_query_text
from lib.search_utils import load_movies


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")
   
    verify_parser = subparsers.add_parser("verify", help="Verify the semantic search model")

    embed_text_parser = subparsers.add_parser("embed_text", help="Embed input text and display the embeddings")
    embed_text_parser.add_argument("text", type=str, nargs='?', help="Text to embed (if not provided, will prompt for input)")

    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="Verify that cached embeddings can be loaded and have correct dimensions")

    embed_query_text_parser = subparsers.add_parser("embedquery", help="Embed a query text and display the embedding")
    embed_query_text_parser.add_argument("query", type=str, nargs='?', help="Query text to embed (if not provided, will prompt for input)")

    search_parser = subparsers.add_parser("search", help="Search for movies semantically similar to the query")         
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--limit", type=int, nargs='?', default=5, help="Maximum number of results to return, default: {default})".format(default=5))
 
    chunk_parser = subparsers.add_parser("chunk", help="Chunk documents into smaller pieces for embedding if needed in the future")
    chunk_parser.add_argument("text", type=str, nargs='?', help="Text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, nargs='?', default=200, help="Size of each chunk, default: {default})".format(default=200))
    chunk_parser.add_argument("--overlap", type=int , nargs='?', default=0, help="ammount of overlap in chunk")

    sem_chunk_parser =subparsers.add_parser("semantic_chunk", help="Semantic chunk documents")
    sem_chunk_parser.add_argument("text", type=str, nargs='?', help="Text to chunk")
    sem_chunk_parser.add_argument("--max-chunk-size", type=int, nargs='?', default=4, help="Max chunk size, defaults to 4")
    sem_chunk_parser.add_argument("--overlap", type=int , nargs='?', default=0, help="ammount of overlap in chunk")


    args = parser.parse_args()

    match args.command:

        case "verify":
            print("Verifying model...")
            verify_command()

        case "embed_text":
            print("Embedding text...")
            embed_text_command(args.text) # if args.text else input("Enter text to embed: "))

        case "verify_embeddings":
            print("Verifying embeddings...")
            verify_embeddings_command()

        case "embedquery":
            print("Embedding query text...")
            embed_query_text_command(args.query)

        case "search":
            print("Searching...")
            search_command(args.query, limit=args.limit)

        case "chunk":
            print("Chunking documents...")
            chunk_command(args.text, args.chunk_size, args.overlap)

        case "semantic_chunk":
            print("Semantically chunking...")
            sem_chunks = semantic_chunk_command(args.text, max_chunk_size=args.max_chunk_size, overlap=args.overlap)
            print_chunks(sem_chunks)

        case _:
            parser.print_help()

def verify_command():
    model = SemanticSearch()
    print(f"Model loaded: {model.model}")
    print(f"Max sequence length: {model.model.max_seq_length}")

def embed_text_command(text):
    model = SemanticSearch()
    embedding = model.generate_embeddings(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings_command():
    verify_embeddings()

def embed_query_text_command(query):
    embed_query_text(query)

def search_command(query, limit=5):
    sem_search = SemanticSearch()
    movie_list = load_movies()
    sem_search.load_or_create_embeddings(movie_list)
    results = sem_search.search(query, limit=limit)

    print(f"Search results for query: '{query}'")
    for i, result in enumerate(results):
        print(f"{i+1}. {result['title']} (Score: {result['score']:.4f})")
        print(f"   {result['description']}\n")

def chunk_command(text, chunk_size=200, overlap=0):
    text_length = len(text)
    split_text = text.split() # split on whitespace
    num_words = len(split_text)

    print(f"Chunking {text_length} characters")

    chunks = []

    start_pos = 0
    end_pos = chunk_size
    remaining = num_words

    while remaining > 0:
        if start_pos < overlap:
            chunk = " ".join(split_text[start_pos:end_pos])
        else:
            chunk = " ".join(split_text[(start_pos-overlap):end_pos])
        used = end_pos - start_pos
        remaining -= used

        start_pos = end_pos
        end_pos += chunk_size
        if end_pos > num_words:
            end_pos = num_words
        
        chunks.append(chunk) 

    print_chunks(chunks)

def semantic_chunk_command(text, max_chunk_size, overlap):
    """    
    Split the input into individual sentences by using a regular expression. 
    The re.split function and this nasty regex should help: r"(?<=[.!?])\s+"
    """
    text_length = len(text)
    print(f"Semantically chunking {text_length} characters")

    split_text = re.split(r"(?<=[.!?])\s+",text)
    tot_num_sentences = len(split_text)


    chunks = []

    start_pos = 0
    end_pos = max_chunk_size
    remaining = tot_num_sentences

    while remaining > 0:
        if start_pos == 0:
            chunk = " ".join(split_text[start_pos:end_pos])
        else:
            chunk = " ".join(split_text[(start_pos-overlap):end_pos])
        used = end_pos - start_pos
        remaining -= used

        start_pos = end_pos
        end_pos += max_chunk_size - overlap
        if end_pos > tot_num_sentences:
            end_pos = tot_num_sentences
        
        chunks.append(chunk) 

    return chunks
                          

    """Each chunk should contain up to max_chunk_size sentences.
Support overlap by number of sentences.
Return a list of chunk strings."""

def print_chunks(chunks):
    for i in range(len(chunks)):
        print(f"{i + 1}. {chunks[i]}")


if __name__ == "__main__":
    main()