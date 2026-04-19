"""
Training script for Conformer-CTC Russian numbers ASR.

Usage:
    python train.py --train_csv data/train.csv \
                    --dev_csv   data/dev.csv   \
                    --data_root data/          \
                    --output_dir checkpoints/  \
                    --epochs 50
"""
from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import NumbersDataset, collate_fn, compute_mean_std
from decode import compute_cer_batch, decode_batch
from model import ConformerCTC
from text_utils import get_vocab, num_to_words


# --------------------------------------------------------------------------- #
#  Helpers                                                                    #
# --------------------------------------------------------------------------- #

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
def evaluate(model, loader, device, idx2char, use_beam=False, beam_width=5):
    """Returns dict: {overall_cer, spk_cer: {spk_id: cer}, sample_outputs}"""
    model.eval()

    refs_text, hyps_text = [], []
    spk_refs: dict[str, list] = {}
    spk_hyps: dict[str, list] = {}

    for batch in loader:
        mel = batch["mel"].to(device)
        mel_len = batch["mel_len"].to(device)

        log_probs, out_len = model(mel, mel_len)
        preds = decode_batch(log_probs.cpu(), out_len.cpu(), idx2char, use_beam=use_beam, beam_width=beam_width)

        for i, (pred_int, true_int) in enumerate(zip(preds, batch["transcription"])):
            ref_str = str(true_int)
            hyp_str = str(pred_int) if pred_int is not None else ""
            refs_text.append(ref_str)
            hyps_text.append(hyp_str)

    overall_cer = compute_cer_batch(refs_text, hyps_text)
    return {
        "cer": overall_cer,
        "sample": list(zip(refs_text[:5], hyps_text[:5])),
    }


# --------------------------------------------------------------------------- #
#  Main                                                                       #
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", default="data/train.csv")
    p.add_argument("--dev_csv", default="data/dev.csv")
    p.add_argument("--data_root", default="data/")
    p.add_argument("--output_dir", default="checkpoints/")
    p.add_argument("--noise_dir", default=None, help="Optional noise folder for augmentation")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup_steps", type=int, default=0)
    p.add_argument("--d_model", type=int, default=144)
    p.add_argument("--n_layers", type=int, default=6)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--resume", default=None, help="checkpoint path to resume from")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--beam_width", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Device: {device}")

    # Vocab
    char2idx, idx2char = get_vocab()
    vocab_size = len(char2idx) + 1   # +1 for blank at index 0
    print(f"Vocab size: {vocab_size}")

    # Datasets
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

    # Model
    model = ConformerCTC(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_len=4096,
    ).to(device)

    n_params = model.count_parameters()
    print(f"Model parameters: {n_params:,} ({n_params/1e6:.2f}M)")
    assert n_params <= 5_000_000, f"Model too large: {n_params} > 5M"

    # Save config
    config = {
        "vocab_size": vocab_size, "d_model": args.d_model,
        "n_layers": args.n_layers, "n_heads": args.n_heads,
        "n_params": n_params,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Loss, optimizer, scheduler
    criterion = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    # Warmup + cosine decay scheduler
    total_steps = args.epochs * len(train_loader)
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return max(0.05, 0.5 * (1.0 + torch.cos(torch.tensor(3.14159 * progress)).item()))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # AMP
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

        eval_result = evaluate(model, dev_loader, device, idx2char, use_beam=False)
        val_cer = eval_result["cer"]

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1:03d}/{args.epochs}  "
            f"loss={train_loss:.4f}  CER={val_cer:.4f}  "
            f"lr={lr_now:.2e}  {elapsed:.1f}s"
        )
        print(f"  Samples: {eval_result['sample']}")

        history.append({"epoch": epoch + 1, "loss": train_loss, "cer": val_cer})

        is_best = val_cer < best_cer
        if is_best:
            best_cer = val_cer
            save_checkpoint(model, optimizer, scheduler, epoch + 1, best_cer,
                            output_dir / "best.pt", mean, std)
            print(f"  ✓ New best CER: {best_cer:.4f}")

        save_checkpoint(model, optimizer, scheduler, epoch + 1, best_cer,
                        output_dir / "last.pt", mean, std)

    print("\nRunning final beam-search evaluation on best checkpoint...")
    load_checkpoint(model, None, None, output_dir / "best.pt")
    final = evaluate(
        model, dev_loader, device, idx2char, use_beam=True, beam_width=args.beam_width
    )
    print(f"Final beam-search CER: {final['cer']:.4f}")

    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nDone! Best CER: {best_cer:.4f}")
    print(f"Checkpoints saved to: {output_dir}")


if __name__ == "__main__":
    main()
