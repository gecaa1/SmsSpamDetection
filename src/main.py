import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Opis programu")
    parser.add_argument("--input", required=False, help="Sciezka do pliku")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # TODO: implementacja algorytmu
    if args.input:
        print(f"Input: {args.input}")


if __name__ == "__main__":
    main()
