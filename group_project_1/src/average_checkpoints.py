"""
Average top-K checkpoints saved by train.py.

Examples:
    python src/average_checkpoints.py --checkpoint-dir checkpoints/checkpoints_v3
    python src/average_checkpoints.py --checkpoint-dir checkpoints/checkpoints_v3 --top-k 3
"""

import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--manifest", default="top_k_checkpoints.json")
    parser.add_argument("--top-k", type=int, default=None, help="Average only the first K entries from manifest")
    parser.add_argument("--output-name", default="averaged.pt")
    return parser.parse_args()


def average_state_dicts(paths: list[Path]) -> dict[str, torch.Tensor]:
    averaged: dict[str, torch.Tensor] = {}
    for idx, path in enumerate(paths):
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint["model_state"]
        if idx == 0:
            averaged = {key: value.detach().float().clone() for key, value in state_dict.items()}
            continue
        for key, value in state_dict.items():
            averaged[key] += value.detach().float()

    scale = 1.0 / max(len(paths), 1)
    for key in averaged:
        averaged[key].mul_(scale)
    return averaged


def main() -> None:
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    manifest_path = checkpoint_dir / args.manifest
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if not manifest:
        raise RuntimeError("Manifest is empty. Train with --save_top_k > 0 first.")

    entries = manifest if args.top_k is None else manifest[: args.top_k]
    paths = [checkpoint_dir / entry["path"] for entry in entries]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint listed in manifest does not exist: {path}")

    averaged = average_state_dicts(paths)
    output_path = checkpoint_dir / args.output_name
    torch.save(
        {
            "model_state": averaged,
            "source_checkpoints": [path.name for path in paths],
        },
        output_path,
    )
    print(f"Averaged {len(paths)} checkpoints -> {output_path}")


if __name__ == "__main__":
    main()
