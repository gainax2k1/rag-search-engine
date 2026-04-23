#!/usr/bin/env python3

import argparse
from lib.semantic_search import SemanticSearch


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")
   
    verify_parser = subparsers.add_parser("verify", help="Verify the semantic search model")
    
    args = parser.parse_args()

    match args.command:

        case "verify":
            print("Verifying model...")
            verify_command()

        case _:
            parser.print_help()



def verify_command():
    model = SemanticSearch()
    print(f"Model loaded: {model.model}")
    print(f"Max sequence length: {model.model.max_seq_length}")

if __name__ == "__main__":
    main()