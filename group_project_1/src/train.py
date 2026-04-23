"""
Training script for Conformer-CTC Russian numbers ASR.

Usage:
    python train.py --train_csv data/train.csv \
                    --dev_csv   data/dev.csv   \
                    --data_root data/          \
                    --output_dir checkpoints/  \
                    --epochs 50
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import NumbersDataset, collate_fn, compute_mean_std
from decode import compute_cer_batch, decode_batch
from lm_utils import KenLMConfig, KenLMScorer
from model import ConformerCTC
from text_utils import get_vocab


# --------------------------------------------------------------------------- #
#  Helpers                                                                    #
# --------------------------------------------------------------------------- #

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(model, optimizer, scheduler, epoch, best_cer, path, mean, std):
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "best_cer": best_cer,
            "mean": mean,
            "std": std,
        },
        path,
    )


def load_checkpoint(model, optimizer, scheduler, path):
    ck = torch.load(path, map_location="cpu")
    model.load_state_dict(ck["model_state"])
    if optimizer and "optimizer_state" in ck:
        optimizer.load_state_dict(ck["optimizer_state"])
    if scheduler and ck.get("scheduler_state"):
        scheduler.load_state_dict(ck["scheduler_state"])
    return ck.get("epoch", 0), ck.get("best_cer", float("inf")), ck.get("mean"), ck.get("std")


def compute_exact_match(references: list[str], hypotheses: list[str]) -> float:
    if not references:
        return 0.0
    return sum(ref == hyp for ref, hyp in zip(references, hypotheses)) / len(references)


def aggregate_group_metrics(
    refs: list[str],
    hyps: list[str],
    groups: list[str | None],
) -> dict[str, dict[str, float | int]]:
    grouped_refs: dict[str, list[str]] = {}
    grouped_hyps: dict[str, list[str]] = {}

    for ref, hyp, group in zip(refs, hyps, groups):
        key = "unknown" if group is None else str(group)
        grouped_refs.setdefault(key, []).append(ref)
        grouped_hyps.setdefault(key, []).append(hyp)

    return {
        key: {
            "samples": len(grouped_refs[key]),
            "cer": compute_cer_batch(grouped_refs[key], grouped_hyps[key]),
            "exact": compute_exact_match(grouped_refs[key], grouped_hyps[key]),
        }
        for key in sorted(grouped_refs)
    }


def sanitize_metric_name(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


class TopKCheckpointManager:
    def __init__(self, output_dir: Path, top_k: int):
        self.output_dir = output_dir
        self.top_k = max(top_k, 0)
        self.entries: list[dict[str, float | int | str]] = []

    def maybe_save(
        self,
        metric_value: float,
        metric_name: str,
        model,
        optimizer,
        scheduler,
        epoch: int,
        best_cer: float,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> None:
        if self.top_k <= 0:
            return

        worst = max(self.entries, key=lambda item: item["metric_value"], default=None)
        if len(self.entries) >= self.top_k and worst is not None and metric_value >= float(worst["metric_value"]):
            return

        filename = f"epoch{epoch:03d}_{sanitize_metric_name(metric_name)}_{metric_value:.6f}.pt"
        path = self.output_dir / filename
        save_checkpoint(model, optimizer, scheduler, epoch, best_cer, path, mean, std)
        self.entries.append(
            {
                "epoch": epoch,
                "metric_name": metric_name,
                "metric_value": metric_value,
                "path": filename,
            }
        )
        self.entries.sort(key=lambda item: float(item["metric_value"]))

        while len(self.entries) > self.top_k:
            removed = self.entries.pop()
            removed_path = self.output_dir / str(removed["path"])
            if removed_path.exists():
                removed_path.unlink()

    def write_manifest(self) -> None:
        with open(self.output_dir / "top_k_checkpoints.json", "w", encoding="utf-8") as f:
            json.dump(self.entries, f, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------- #
#  Training loop                                                              #
# --------------------------------------------------------------------------- #

def train_epoch(model, loader, optimizer, criterion, device, scaler=None, scheduler=None):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        mel = batch["mel"].to(device)                 # (B, T, F)
        mel_len = batch["mel_len"].to(device)
        labels = batch["label"].to(device)            # (B, L)
        label_len = batch["label_len"].to(device)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                log_probs, out_len = model(mel, mel_len)
                # CTC loss: log_probs (T, B, C), targets (sum of labels)
                targets = torch.cat([labels[i, :label_len[i]] for i in range(len(label_len))])
                loss = criterion(log_probs, targets, out_len, label_len)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            log_probs, out_len = model(mel, mel_len)
            targets = torch.cat([labels[i, :label_len[i]] for i in range(len(label_len))])
            loss = criterion(log_probs, targets, out_len, label_len)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    idx2char,
    use_beam=False,
    beam_width=5,
    kenlm_scorer=None,
    lm_top_paths=None,
):
    """Run validation in the same decoding mode used for inference."""
    model.eval()

    refs_text, hyps_text = [], []
    speaker_groups: list[str | None] = []
    ext_groups: list[str | None] = []
    parse_failures = 0

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

        for i, (pred_int, true_int) in enumerate(zip(preds, batch["transcription"])):
            ref_str = str(true_int)
            hyp_str = str(pred_int) if pred_int is not None else ""
            refs_text.append(ref_str)
            hyps_text.append(hyp_str)
            speaker_groups.append(batch["spk_id"][i] if "spk_id" in batch else None)
            ext_groups.append(batch["ext"][i] if "ext" in batch else None)
            if pred_int is None:
                parse_failures += 1

    overall_cer = compute_cer_batch(refs_text, hyps_text)
    exact = compute_exact_match(refs_text, hyps_text)
    return {
        "cer": overall_cer,
        "exact": exact,
        "num_samples": len(refs_text),
        "parse_failures": parse_failures,
        "parse_failure_rate": parse_failures / max(len(refs_text), 1),
        "speaker_metrics": aggregate_group_metrics(refs_text, hyps_text, speaker_groups),
        "ext_metrics": aggregate_group_metrics(refs_text, hyps_text, ext_groups),
        "sample": list(zip(refs_text[:5], hyps_text[:5])),
    }


# --------------------------------------------------------------------------- #
#  Main                                                                       #
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", default="/data/tsa/itmo/ASR_TTS/group_project_1/data/train.csv")
    p.add_argument("--dev_csv", default="/data/tsa/itmo/ASR_TTS/group_project_1/data/dev.csv")
    p.add_argument("--data_root", default="/data/tsa/itmo/ASR_TTS/group_project_1/data/")
    p.add_argument("--output_dir", default="/data/tsa/itmo/ASR_TTS/group_project_1/checkpoints/")
    p.add_argument("--noise_dir", default=None, help="Optional noise folder for augmentation")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--warmup_steps", type=int, default=0)
    p.add_argument("--model_preset", choices=["baseline", "deep8", "wide6"], default="baseline")
    p.add_argument("--d_model", type=int, default=144)
    p.add_argument("--n_layers", type=int, default=6)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--conv_kernel", type=int, default=31)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--resume", default=None, help="checkpoint path to resume from")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--beam_width", type=int, default=5)
    p.add_argument("--kenlm_model", default="/data/tsa/itmo/ASR_TTS/group_project_1/data/lm/numbers-3gram.binary" , help="Optional KenLM ARPA/bin model for beam rescoring")
    p.add_argument("--lm_weight", type=float, default=0.3, help="KenLM weight for beam rescoring")
    p.add_argument("--word_score", type=float, default=0.0, help="Word insertion bonus for KenLM rescoring")
    p.add_argument("--lm_top_paths", type=int, default=8, help="How many beam candidates to rescore with KenLM")
    p.add_argument("--select_metric", choices=["greedy_cer", "decode_cer"], default="decode_cer")
    p.add_argument("--save_top_k", type=int, default=5, help="Number of best checkpoints to retain for averaging")
    return p.parse_args()


def apply_model_preset(args) -> None:
    presets = {
        "baseline": {"d_model": 144, "n_layers": 6, "n_heads": 4, "conv_kernel": 31, "dropout": 0.1},
        "deep8": {"d_model": 144, "n_layers": 8, "n_heads": 4, "conv_kernel": 31, "dropout": 0.1},
        "wide6": {"d_model": 160, "n_layers": 6, "n_heads": 4, "conv_kernel": 31, "dropout": 0.1},
    }
    preset = presets[args.model_preset]
    defaults = {
        "d_model": 144,
        "n_layers": 6,
        "n_heads": 4,
        "conv_kernel": 31,
        "dropout": 0.1,
    }
    for key, default_value in defaults.items():
        if getattr(args, key) == default_value:
            setattr(args, key, preset[key])


def main():
    args = parse_args()
    apply_model_preset(args)
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    topk_manager = TopKCheckpointManager(output_dir, args.save_top_k)

    device = get_device()
    print(f"Device: {device}")

    kenlm_scorer = None
    if args.kenlm_model:
        kenlm_scorer = KenLMScorer(
            KenLMConfig(
                model_path=args.kenlm_model,
                alpha=args.lm_weight,
                beta=args.word_score,
            )
        )
        print(f"Loaded KenLM model: {args.kenlm_model}")

    char2idx, idx2char = get_vocab()
    vocab_size = len(char2idx) + 1   # +1 for blank at index 0
    print(f"Vocab size: {vocab_size}")

    stats_ds = NumbersDataset(args.train_csv, args.data_root, augment=False)
    train_ds = NumbersDataset(args.train_csv, args.data_root, augment=True, noise_dir=args.noise_dir)
    dev_ds = NumbersDataset(args.dev_csv, args.data_root, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True,
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers,
    )

    model = ConformerCTC(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        conv_kernel=args.conv_kernel,
        dropout=args.dropout,
        max_len=4096,
    ).to(device)

    n_params = model.count_parameters()
    print(f"Model parameters: {n_params:,} ({n_params/1e6:.2f}M)")
    assert n_params <= 5_000_000, f"Model too large: {n_params} > 5M"

    config = {
        "seed": args.seed,
        "model_preset": args.model_preset,
        "vocab_size": vocab_size, "d_model": args.d_model,
        "n_layers": args.n_layers, "n_heads": args.n_heads,
        "conv_kernel": args.conv_kernel, "dropout": args.dropout,
        "n_params": n_params,
        "train_args": vars(args),
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    criterion = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = args.epochs * len(train_loader)
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return max(0.05, 0.5 * (1.0 + torch.cos(torch.tensor(3.14159 * progress)).item()))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    use_amp = not args.no_amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    start_epoch = 0
    best_cer = float("inf")

    if args.resume:
        start_epoch, best_cer, saved_mean, saved_std = load_checkpoint(
            model, optimizer, scheduler, args.resume
        )
        mean = saved_mean
        std = saved_std
        print(f"Resumed from epoch {start_epoch}, best CER={best_cer:.4f}")
    else:
        print("Computing mel normalization stats...")
        mean, std = compute_mean_std(stats_ds)

    if mean is None or std is None:
        raise RuntimeError("Failed to initialize mel normalization stats.")

    torch.save({"mean": mean, "std": std}, output_dir / "norm_stats.pt")
    train_ds.mel_mean = mean
    train_ds.mel_std = std
    dev_ds.mel_mean = mean
    dev_ds.mel_std = std

    print("\n" + "="*60)
    print("Training started")
    print("="*60)

    history = []
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler, scheduler
        )

        greedy_result = evaluate(model, dev_loader, device, idx2char, use_beam=False)
        decode_result = evaluate(
            model,
            dev_loader,
            device,
            idx2char,
            use_beam=True,
            beam_width=args.beam_width,
            kenlm_scorer=kenlm_scorer,
            lm_top_paths=args.lm_top_paths,
        )
        selection_metric = decode_result["cer"] if args.select_metric == "decode_cer" else greedy_result["cer"]

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1:03d}/{args.epochs}  "
            f"loss={train_loss:.4f}  "
            f"greedy_CER={greedy_result['cer']:.4f}  "
            f"decode_CER={decode_result['cer']:.4f}  "
            f"lr={lr_now:.2e}  {elapsed:.1f}s"
        )
        print(
            f"  exact(greedy/decode)=({greedy_result['exact']:.4f}/{decode_result['exact']:.4f})  "
            f"parse_fail={decode_result['parse_failures']}/{decode_result['num_samples']}"
        )
        print(f"  Samples: {decode_result['sample']}")

        history_entry = {
            "epoch": epoch + 1,
            "loss": train_loss,
            "lr": lr_now,
            "greedy": greedy_result,
            "decode": decode_result,
            "selection_metric": args.select_metric,
            "selection_value": selection_metric,
        }
        history.append(history_entry)

        is_best = selection_metric < best_cer
        if is_best:
            best_cer = selection_metric
            save_checkpoint(model, optimizer, scheduler, epoch + 1, best_cer,
                            output_dir / "best.pt", mean, std)
            print(f"  ✓ New best {args.select_metric}: {best_cer:.4f}")

        topk_manager.maybe_save(
            metric_value=float(selection_metric),
            metric_name=args.select_metric,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch + 1,
            best_cer=best_cer,
            mean=mean,
            std=std,
        )

        save_checkpoint(model, optimizer, scheduler, epoch + 1, best_cer,
                        output_dir / "last.pt", mean, std)
        topk_manager.write_manifest()

    print("\nRunning final beam-search evaluation on best checkpoint...")
    load_checkpoint(model, None, None, output_dir / "best.pt")
    final = evaluate(
        model,
        dev_loader,
        device,
        idx2char,
        use_beam=True,
        beam_width=args.beam_width,
        kenlm_scorer=kenlm_scorer,
        lm_top_paths=args.lm_top_paths,
    )
    print(f"Final beam-search CER: {final['cer']:.4f}")
    print(f"Final beam-search exact: {final['exact']:.4f}")

    with open(output_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print(f"\nDone! Best CER: {best_cer:.4f}")
    print(f"Checkpoints saved to: {output_dir}")


if __name__ == "__main__":
    main()
