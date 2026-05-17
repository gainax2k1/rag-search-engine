import argparse
from lib.multimodal_search import MultimodalSearch, verify_image_embedding

def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")


    normalize_parser = subparsers.add_parser("verify_image_embedding", help="Verify that the image embedding can be generated and has the expected dimensions")
    normalize_parser.add_argument("image_path", type=str, help="Path to the image file")


    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            print("Verifying image embedding...")
            verify_image_embedding(args.image_path)

if __name__ == "__main__":
    main()
