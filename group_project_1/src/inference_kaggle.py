"""
Inference CLI for local validation and Kaggle submission generation.

Examples:
    python src/inference_kaggle.py --split dev --output-dir outputs/dev_run
    python src/inference_kaggle.py --split test --output-dir outputs/test_run
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset

from dataset import TARGET_SR, wav_to_mel
from decode import compute_cer_batch, decode_batch
from model import ConformerCTC
from text_utils import get_vocab


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class InferenceDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        data_root: str | Path,
        mel_mean: torch.Tensor,
        mel_std: torch.Tensor,
    ):
        self.df = pd.read_csv(csv_path)
        self.data_root = Path(data_root)
        self.mel_mean = mel_mean
        self.mel_std = mel_std

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        audio_path = self.data_root / row["filename"]
        waveform, sr = torchaudio.load(str(audio_path))
        mel = wav_to_mel(waveform, sr)
        mel = (mel - self.mel_mean) / self.mel_std

        item = {
            "mel": mel,
            "mel_len": mel.shape[0],
            "filename": row["filename"],
        }
        if "transcription" in row.index and pd.notna(row["transcription"]):
            item["transcription"] = int(row["transcription"])
        return item


def inference_collate_fn(batch: list[dict]) -> dict:
    batch.sort(key=lambda x: x["mel_len"], reverse=True)

    mel_lens = torch.tensor([b["mel_len"] for b in batch], dtype=torch.long)
    max_mel = mel_lens[0].item()
    n_mels = batch[0]["mel"].shape[1]

    mels = torch.zeros(len(batch), max_mel, n_mels)
    for i, sample in enumerate(batch):
        m_len = sample["mel_len"]
        mels[i, :m_len] = sample["mel"]

    result = {
        "mel": mels,
        "mel_len": mel_lens,
        "filename": [b["filename"] for b in batch],
    }
    if "transcription" in batch[0]:
        result["transcription"] = [b["transcription"] for b in batch]
    return result


def load_model(checkpoint_dir: Path, device: torch.device) -> tuple[ConformerCTC, torch.Tensor, torch.Tensor]:
    with open(checkpoint_dir / "config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    model = ConformerCTC(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        max_len=4096,
        dropout=0.0,
    ).to(device)

    checkpoint = torch.load(checkpoint_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    norm_stats = torch.load(checkpoint_dir / "norm_stats.pt", map_location="cpu")
    mel_mean = norm_stats["mean"].float()
    mel_std = norm_stats["std"].float()
    return model, mel_mean, mel_std


@torch.no_grad()
def run_inference(
    model: ConformerCTC,
    loader: DataLoader,
    device: torch.device,
    idx2char: dict[int, str],
    use_beam: bool,
    beam_width: int,
) -> tuple[list[dict], dict[str, float | int]]:
    rows: list[dict] = []
    refs: list[str] = []
    hyps: list[str] = []

    for batch in loader:
        mel = batch["mel"].to(device)
        mel_len = batch["mel_len"].to(device)

        log_probs, out_len = model(mel, mel_len)
        preds = decode_batch(
            log_probs.cpu(),
            out_len.cpu(),
            idx2char,
            use_beam=use_beam,
            beam_width=beam_width,
        )

        for i, pred in enumerate(preds):
            pred = 1000 if pred is None else int(max(1000, min(999999, pred)))
            row = {
                "filename": batch["filename"][i],
                "prediction": pred,
            }
            if "transcription" in batch:
                truth = int(batch["transcription"][i])
                row["transcription"] = truth
                row["reference_str"] = str(truth)
                row["prediction_str"] = str(pred)
                refs.append(row["reference_str"])
                hyps.append(row["prediction_str"])
            rows.append(row)

    metrics: dict[str, float | int] = {"num_samples": len(rows)}
    if refs:
        metrics["cer"] = compute_cer_batch(refs, hyps)
    return rows, metrics


def save_outputs(rows: list[dict], split: str, output_dir: Path, metrics: dict[str, float | int]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_df = pd.DataFrame(rows)
    predictions_path = output_dir / f"{split}_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)

    submission_df = predictions_df[["filename", "prediction"]].rename(columns={"prediction": "transcription"})
    submission_df["transcription"] = submission_df["transcription"].astype(str).str.replace(r"\D+", "", regex=True)
    submission_df["transcription"] = submission_df["transcription"].apply(
        lambda x: str(min(999999, max(1000, int(x)))) if x else "1000"
    )
    submission_path = output_dir / "submission.csv"
    submission_df.to_csv(submission_path, index=False)

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--checkpoint-dir", default="src/checkpoints")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--beam", action="store_true", help="Use beam search decoding instead of greedy")
    parser.add_argument("--beam-width", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    _, idx2char = get_vocab()

    checkpoint_dir = Path(args.checkpoint_dir)
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    csv_path = data_root / f"{args.split}.csv"

    model, mel_mean, mel_std = load_model(checkpoint_dir, device)
    dataset = InferenceDataset(csv_path, data_root, mel_mean, mel_std)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=inference_collate_fn,
    )

    rows, metrics = run_inference(
        model=model,
        loader=loader,
        device=device,
        idx2char=idx2char,
        use_beam=args.beam,
        beam_width=args.beam_width,
    )
    save_outputs(rows, args.split, output_dir, metrics)

    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"Split: {args.split}")
    print(f"Samples: {metrics['num_samples']}")
    if "cer" in metrics:
        print(f"CER: {metrics['cer']:.6f}")
    print(f"Saved predictions to: {output_dir / f'{args.split}_predictions.csv'}")
    print(f"Saved submission to: {output_dir / 'submission.csv'}")


if __name__ == "__main__":
    main()
