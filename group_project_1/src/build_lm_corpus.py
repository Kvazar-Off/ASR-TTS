"""
Export a KenLM training corpus from train.csv

Examples:
    python src/build_lm_corpus.py --train-csv data/train.csv --output data/lm/train.txt
"""

import argparse
import csv
from pathlib import Path

from text_utils import num_to_words


def read_transcriptions_from_csv(csv_path: Path) -> list[int]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [int(row["transcription"]) for row in reader]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", default=None, help="Path to extracted train.csv")
    parser.add_argument("--output", required=True, help="Where to save the plain-text LM corpus")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.train_csv:
        transcriptions = read_transcriptions_from_csv(Path(args.train_csv))
    else:
        raise ValueError("Specify either --train-csv or --dataset-zip")

    with output_path.open("w", encoding="utf-8", newline="\n") as f:
        for value in transcriptions:
            f.write(num_to_words(value))
            f.write("\n")

    print(f"Saved {len(transcriptions)} lines to {output_path}")
    print("You can train KenLM on this corpus with an external lmplz/build_binary pipeline.")


if __name__ == "__main__":
    main()
