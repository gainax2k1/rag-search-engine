import argparse
from lib.multimodal_search import MultimodalSearch, verify_image_embedding, image_search_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")


    normalize_parser = subparsers.add_parser("verify_image_embedding", help="Verify that the image embedding can be generated and has the expected dimensions")
    normalize_parser.add_argument("image_path", type=str, help="Path to the image file")

    image_search_parser = subparsers.add_parser("image_search", help="Perform image search against movie database")
    image_search_parser.add_argument("image_path", type=str, help="Path to the image file to search with")


    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            print("Verifying image embedding...")
            verify_image_embedding(args.image_path)

        case "image_search":
            print("Performing image search...")
            image_search_command(args.image_path, limit=5)

        case _:
            parser.print_help()
if __name__ == "__main__":
    main()
