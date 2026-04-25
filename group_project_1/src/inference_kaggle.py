"""
Inference CLI for local validation and Kaggle submission generation.

Examples:
    python src/inference_kaggle.py --split test --output-dir outputs/test_run
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset

from dataset import TARGET_SR, load_audio, wav_to_mel
from decode import compute_cer_batch, decode_batch
from lm_utils import KenLMConfig, KenLMScorer
from model import ConformerCTC
from text_utils import get_vocab


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_exact_match(references: list[str], hypotheses: list[str]) -> float:
    if not references:
        return 0.0
    return sum(ref == hyp for ref, hyp in zip(references, hypotheses)) / len(references)


def aggregate_group_metrics(rows: list[dict], group_key: str) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        if "transcription" not in row:
            continue
        key = row.get(group_key) or "unknown"
        grouped.setdefault(str(key), []).append(row)

    metrics: dict[str, dict[str, float | int]] = {}
    for key, samples in grouped.items():
        refs = [str(item["transcription"]) for item in samples]
        hyps = [str(item["prediction"]) for item in samples]
        metrics[key] = {
            "samples": len(samples),
            "cer": compute_cer_batch(refs, hyps),
            "exact": compute_exact_match(refs, hyps),
        }
    return metrics


class InferenceDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        data_root: str | Path,
        mel_mean: torch.Tensor | None,
        mel_std: torch.Tensor | None,
        cmvn_mode: str = "global",
    ):
        if cmvn_mode not in {"utterance", "global"}:
            raise ValueError(f"Unknown cmvn_mode: {cmvn_mode}")
        self.df = pd.read_csv(csv_path)
        self.data_root = Path(data_root)
        self.mel_mean = mel_mean
        self.mel_std = mel_std
        self.cmvn_mode = cmvn_mode

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        audio_path = self.data_root / row["filename"]
        waveform, sr = load_audio(audio_path)
        mel = wav_to_mel(waveform, sr)
        if self.cmvn_mode == "utterance":
            mean = mel.mean(dim=0, keepdim=True)
            std = mel.std(dim=0, keepdim=True).clamp(min=1e-5)
            mel = (mel - mean) / std
        else:
            mel = (mel - self.mel_mean) / self.mel_std

        item = {
            "mel": mel,
            "mel_len": mel.shape[0],
            "filename": row["filename"],
            "spk_id": row["spk_id"] if "spk_id" in row.index else None,
            "gender": row["gender"] if "gender" in row.index else None,
            "ext": row["ext"] if "ext" in row.index else audio_path.suffix.lstrip("."),
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
        "spk_id": [b.get("spk_id") for b in batch],
        "gender": [b.get("gender") for b in batch],
        "ext": [b.get("ext") for b in batch],
    }
    if "transcription" in batch[0]:
        result["transcription"] = [b["transcription"] for b in batch]
    return result


def load_model(
    checkpoint_dir: Path,
    device: torch.device,
    checkpoint_name: str,
) -> tuple[ConformerCTC, torch.Tensor | None, torch.Tensor | None, str]:
    with open(checkpoint_dir / "config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    model = ConformerCTC(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        conv_kernel=cfg.get("conv_kernel", 31),
        max_len=4096,
        dropout=cfg.get("dropout", 0.0),
    ).to(device)

    checkpoint = torch.load(checkpoint_dir / checkpoint_name, map_location=device)
    state_dict = checkpoint.get("model_state", checkpoint.get("model", checkpoint))
    model.load_state_dict(state_dict)
    model.eval()

    cmvn_mode = cfg.get("cmvn_mode", "global")
    mel_mean: torch.Tensor | None = None
    mel_std: torch.Tensor | None = None

    norm_path = checkpoint_dir / "norm_stats.pt"
    if norm_path.exists():
        norm_stats = torch.load(norm_path, map_location="cpu")
        if "mean" in norm_stats and "std" in norm_stats:
            mel_mean = norm_stats["mean"].float()
            mel_std = norm_stats["std"].float()
        if "cmvn_mode" in norm_stats and "cmvn_mode" not in cfg:
            cmvn_mode = norm_stats["cmvn_mode"]

    if cmvn_mode == "global" and (mel_mean is None or mel_std is None):
        raise RuntimeError("Global CMVN requested but norm_stats.pt is missing mean/std.")

    return model, mel_mean, mel_std, cmvn_mode


@torch.no_grad()
def run_inference(
    model: ConformerCTC,
    loader: DataLoader,
    device: torch.device,
    idx2char: dict[int, str],
    use_beam: bool,
    beam_width: int,
    kenlm_scorer: KenLMScorer | None = None,
    lm_top_paths: int | None = None,
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
            kenlm_scorer=kenlm_scorer,
            lm_top_paths=lm_top_paths,
        )

        for i, pred in enumerate(preds):
            pred = 1000 if pred is None else int(max(1000, min(999999, pred)))
            row = {
                "filename": batch["filename"][i],
                "prediction": pred,
                "spk_id": batch["spk_id"][i] if "spk_id" in batch else None,
                "gender": batch["gender"][i] if "gender" in batch else None,
                "ext": batch["ext"][i] if "ext" in batch else None,
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
        metrics["exact"] = compute_exact_match(refs, hyps)
        metrics["speaker_metrics"] = aggregate_group_metrics(rows, "spk_id")
        metrics["ext_metrics"] = aggregate_group_metrics(rows, "ext")
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
    parser.add_argument("--data-root", default="/data/tsa/itmo/ASR_TTS/group_project_1/data")
    parser.add_argument("--checkpoint-dir", default="/data/tsa/itmo/ASR_TTS/group_project_1/src/checkpoints")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint-name", default="best.pt", help="Checkpoint file inside checkpoint-dir")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--beam", action="store_true", help="Use beam search decoding instead of greedy")
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--kenlm-model", default="/data/tsa/itmo/ASR_TTS/group_project_1/data/lm/numbers-3gram.binary", help="Optional KenLM ARPA/bin model for beam rescoring")
    parser.add_argument("--lm-weight", type=float, default=0.5, help="KenLM weight for beam rescoring")
    parser.add_argument("--word-score", type=float, default=0.0, help="Word insertion bonus for KenLM rescoring")
    parser.add_argument("--lm-top-paths", type=int, default=8, help="How many beam candidates to rescore with KenLM")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    _, idx2char = get_vocab()

    checkpoint_dir = Path(args.checkpoint_dir)
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    csv_path = data_root / f"{args.split}.csv"

    kenlm_scorer = None
    if args.kenlm_model:
        kenlm_scorer = KenLMScorer(
            KenLMConfig(
                model_path=args.kenlm_model,
                alpha=args.lm_weight,
                beta=args.word_score,
            )
        )

    model, mel_mean, mel_std, cmvn_mode = load_model(checkpoint_dir, device, args.checkpoint_name)
    dataset = InferenceDataset(csv_path, data_root, mel_mean, mel_std, cmvn_mode=cmvn_mode)
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
        use_beam=args.beam or kenlm_scorer is not None,
        beam_width=args.beam_width,
        kenlm_scorer=kenlm_scorer,
        lm_top_paths=args.lm_top_paths,
    )
    save_outputs(rows, args.split, output_dir, metrics)

    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_dir / args.checkpoint_name}")
    print(f"CMVN mode: {cmvn_mode}")
    print(f"Split: {args.split}")
    print(f"Samples: {metrics['num_samples']}")
    if "cer" in metrics:
        print(f"CER: {metrics['cer']:.6f}")
        print(f"Exact: {metrics['exact']:.6f}")
    print(f"Saved predictions to: {output_dir / f'{args.split}_predictions.csv'}")
    print(f"Saved submission to: {output_dir / 'submission.csv'}")


if __name__ == "__main__":
    main()
