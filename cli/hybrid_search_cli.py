import argparse

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")


    normalize_parser = subparsers.add_parser("normalize", help="Normalize list of scores")
    normalize_parser.add_argument("scores", nargs="*", type=float, help="list of scores to normalize")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            print("Normalizing scores...")
            normalize_command(args.scores)

        case _:
            parser.print_help()



def normalize_command(scores):
    if len(scores) == 0:
        return
    min_val = min(scores)
    max_val = max(scores)
    if min_val == max_val:
        for score in scores:
            print(f"* 1.0")
        return
    
    norm_scores = []

    for score in scores:
        normed = (score- min_val) / (max_val-min_val)
        norm_scores.append(normed)
    
    for score in norm_scores:
        print(f"* {score:.4f}")




if __name__ == "__main__":
    main()