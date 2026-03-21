"""
run_experiments.py — Full experiment runner for ASR Decoding Assignment 2.

implement tasks: HF_HUB_OFFLINE=1 nohup python run_experiments.py >> nohup_final.out 2>&1 &

Tasks 1-9:
  Task 1  — Greedy decode, eval on LibriSpeech test-other
  Task 2  — Beam search, beam_width sweep
  Task 3  — Temperature sweep (greedy, LibriSpeech test-other)
  Task 4  — Beam + 3-gram LM (shallow fusion), alpha/beta sweep
  Task 5  — Beam + 4-gram LM
  Task 6  — LM rescoring, alpha/beta sweep + qualitative comparison
  Task 7  — All methods on Earnings22, temperature sweep with LM (7b)
  Task 8  — Train financial KenLM (bash commands printed)
  Task 9  — All LMs on both test sets, bar chart

Usage:
    python run_experiments.py [--task TASK] [--quick]

    --task N   run only Task N (1-9); omit to run all
    --quick    use only first 20 samples for fast debugging
"""

import argparse
import os
import csv
import sys
import time
from typing import List, Tuple, Dict, Optional

import kenlm
import torch
import soundfile as sf
import numpy as np
import jiwer
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wav2vec2decoder import Wav2Vec2Decoder


# ---------------------------------------------------------------------------
# LM cache — load each KenLM model only once
# ---------------------------------------------------------------------------

_lm_cache: Dict[str, kenlm.Model] = {}

def get_lm(path: Optional[str]) -> Optional[kenlm.Model]:
    """Return a cached KenLM model (or None)."""
    if path is None:
        return None
    if path not in _lm_cache:
        _lm_cache[path] = kenlm.Model(path)
    return _lm_cache[path]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LM_3GRAM       = "lm/3-gram.pruned.1e-7.arpa.gz"
LM_4GRAM       = "lm/4-gram.arpa.gz"
LM_FINANCIAL   = "lm/financial-3gram.arpa.gz"

LIBRI_MANIFEST  = "data/librispeech_test_other/manifest.csv"
EARNINGS_MANIFEST = "data/earnings22_test/manifest.csv"

RESULTS_DIR = "results"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_manifest(csv_path: str, limit: Optional[int] = None) -> List[Tuple[str, str]]:
    """Return list of (audio_path, reference_text) from a manifest CSV."""
    samples = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit and i >= limit:
                break
            samples.append((row["path"], row["text"].strip()))
    return samples


def compute_metrics(refs: List[str], hyps: List[str]) -> Dict[str, float]:
    """Return WER and CER (as fractions, e.g. 0.104 = 10.4%)."""
    wer = jiwer.wer(refs, hyps)
    cer = jiwer.cer(refs, hyps)
    return {"wer": wer, "cer": cer}


def precompute_logits(
    decoder: Wav2Vec2Decoder,
    samples: List[Tuple[str, str]],
) -> List[torch.Tensor]:
    """Run acoustic model once per sample, return list of raw logits tensors."""
    logits_list = []
    for i, (path, _) in enumerate(samples):
        audio_np, sr = sf.read(path, dtype="float32")
        assert sr == 16000, f"Expected 16kHz, got {sr} for {path}"
        inputs = decoder.processor(
            torch.tensor(audio_np), return_tensors="pt", sampling_rate=16000
        )
        with torch.no_grad():
            logits = decoder.model(inputs.input_values).logits[0]
        logits_list.append(logits)
        if (i + 1) % 20 == 0:
            print(f"    [{i+1}/{len(samples)}] logits computed", flush=True)
    return logits_list


def decode_logits(
    decoder: Wav2Vec2Decoder,
    logits: torch.Tensor,
    method: str,
    temperature: float = 1.0,
) -> str:
    """Apply temperature, then call the appropriate decoder method."""
    scaled = logits / temperature
    if method == "greedy":
        return decoder.greedy_decode(scaled)
    elif method == "beam":
        return decoder.beam_search_decode(scaled)
    elif method == "beam_lm":
        return decoder.beam_search_with_lm(scaled)
    elif method == "beam_lm_rescore":
        beams = decoder.beam_search_decode(scaled, return_beams=True)
        return decoder.lm_rescore(beams)
    else:
        raise ValueError(f"Unknown method '{method}'")


def evaluate_cached(
    decoder: Wav2Vec2Decoder,
    samples: List[Tuple[str, str]],
    logits_list: List[torch.Tensor],
    method: str,
    temperature: float = 1.0,
) -> Tuple[Dict[str, float], List[str]]:
    """Decode from cached logits, return metrics dict and list of hypotheses."""
    refs, hyps = [], []
    for i, ((path, ref), logits) in enumerate(zip(samples, logits_list)):
        hyp = decode_logits(decoder, logits, method, temperature)
        refs.append(ref)
        hyps.append(hyp)
        if (i + 1) % 20 == 0:
            print(f"    [{i+1}/{len(samples)}] done", flush=True)
    metrics = compute_metrics(refs, hyps)
    return metrics, hyps


def print_metrics(label: str, metrics: Dict[str, float]):
    print(f"  {label}: WER={metrics['wer']:.2%}  CER={metrics['cer']:.2%}")


def save_csv(path: str, rows: List[dict]):
    if not rows:
        return
    ensure_dir(os.path.dirname(path))
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved -> {path}")


# ---------------------------------------------------------------------------
# Task 1 — Greedy decoding on LibriSpeech test-other
# ---------------------------------------------------------------------------

def task1(decoder, samples, logits_list):
    print("\n" + "="*60)
    print("TASK 1 — Greedy decoding (LibriSpeech test-other)")
    print("="*60)
    metrics, hyps = evaluate_cached(decoder, samples, logits_list, method="greedy")
    print_metrics("Greedy", metrics)
    rows = [{"path": p, "ref": r, "hyp": h}
            for (p, r), h in zip(samples, hyps)]
    save_csv(f"{RESULTS_DIR}/task1_greedy.csv", rows)
    save_csv(f"{RESULTS_DIR}/task1_metrics.csv",
             [{"method": "greedy", "dataset": "librispeech",
               "wer": metrics["wer"], "cer": metrics["cer"]}])
    return metrics


# ---------------------------------------------------------------------------
# Task 2 — Beam search + beam_width sweep
# ---------------------------------------------------------------------------

def task2(decoder, samples, logits_list):
    print("\n" + "="*60)
    print("TASK 2 — Beam search (LibriSpeech test-other)")
    print("="*60)

    decoder.beam_width = 10
    metrics, hyps = evaluate_cached(decoder, samples, logits_list, method="beam")
    print_metrics("Beam (width=10)", metrics)
    rows = [{"path": p, "ref": r, "hyp": h}
            for (p, r), h in zip(samples, hyps)]
    save_csv(f"{RESULTS_DIR}/task2_beam.csv", rows)

    print("\n  --- beam_width sweep ---")
    sweep_rows = []
    for bw in [1, 3, 10, 50]:
        print(f"  beam_width={bw} ...", flush=True)
        decoder.beam_width = bw
        m, _ = evaluate_cached(decoder, samples, logits_list, method="beam")
        print_metrics(f"  width={bw}", m)
        sweep_rows.append({"beam_width": bw, "wer": m["wer"], "cer": m["cer"]})

    save_csv(f"{RESULTS_DIR}/task2_beam_width_sweep.csv", sweep_rows)

    df = pd.DataFrame(sweep_rows)
    fig, ax = plt.subplots()
    ax.plot(df["beam_width"], df["wer"] * 100, marker="o", label="WER")
    ax.plot(df["beam_width"], df["cer"] * 100, marker="s", label="CER")
    ax.set_xlabel("Beam Width")
    ax.set_ylabel("Error Rate (%)")
    ax.set_title("Task 2 — Beam Width vs Error Rate (LibriSpeech)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    path = f"{RESULTS_DIR}/task2_beam_width_sweep.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved -> {path}")
    return metrics


# ---------------------------------------------------------------------------
# Task 3 — Temperature sweep (greedy, LibriSpeech)
# ---------------------------------------------------------------------------

def task3(decoder, samples, logits_list):
    print("\n" + "="*60)
    print("TASK 3 — Temperature sweep, greedy (LibriSpeech test-other)")
    print("="*60)

    temps = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    rows = []
    for T in temps:
        print(f"  T={T} ...", flush=True)
        m, _ = evaluate_cached(decoder, samples, logits_list,
                               method="greedy", temperature=T)
        print_metrics(f"  T={T}", m)
        rows.append({"temperature": T, "wer": m["wer"], "cer": m["cer"]})

    save_csv(f"{RESULTS_DIR}/task3_temperature_sweep.csv", rows)

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots()
    ax.plot(df["temperature"], df["wer"] * 100, marker="o", label="WER")
    ax.plot(df["temperature"], df["cer"] * 100, marker="s", label="CER")
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Error Rate (%)")
    ax.set_title("Task 3 — Temperature vs Error Rate, Greedy (LibriSpeech)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    path = f"{RESULTS_DIR}/task3_temperature_sweep.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved -> {path}")


# ---------------------------------------------------------------------------
# Task 4 — Beam + 3-gram LM shallow fusion, alpha/beta sweep
# ---------------------------------------------------------------------------

def task4(decoder, samples, logits_list):
    print("\n" + "="*60)
    print("TASK 4 — Beam + 3-gram LM shallow fusion (LibriSpeech)")
    print("="*60)

    decoder.lm_model = get_lm(LM_3GRAM)
    decoder.beam_width = 10

    alphas = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    betas  = [0.0, 0.5, 1.0, 1.5]

    rows = []
    best_wer = float("inf")
    best_alpha, best_beta = 0.1, 0.5

    for alpha in alphas:
        for beta in betas:
            print(f"  alpha={alpha}, beta={beta} ...", flush=True)
            decoder.alpha = alpha
            decoder.beta = beta
            m, _ = evaluate_cached(decoder, samples, logits_list, method="beam_lm")
            print_metrics(f"    alpha={alpha} beta={beta}", m)
            rows.append({"alpha": alpha, "beta": beta,
                         "wer": m["wer"], "cer": m["cer"]})
            if m["wer"] < best_wer:
                best_wer = m["wer"]
                best_alpha, best_beta = alpha, beta

    save_csv(f"{RESULTS_DIR}/task4_lm_sweep.csv", rows)
    print(f"\n  Best config: alpha={best_alpha}, beta={best_beta}, WER={best_wer:.2%}")

    # Heatmap (WER)
    df = pd.DataFrame(rows)
    pivot = df.pivot(index="alpha", columns="beta", values="wer") * 100
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(b) for b in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(a) for a in pivot.index])
    ax.set_xlabel("beta")
    ax.set_ylabel("alpha")
    ax.set_title("Task 4 - WER (%) Heatmap: 3-gram LM Shallow Fusion")
    plt.colorbar(im, ax=ax, label="WER (%)")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f"{pivot.values[i, j]:.1f}",
                    ha="center", va="center", fontsize=7)
    fig.tight_layout()
    path = f"{RESULTS_DIR}/task4_lm_sweep_heatmap.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved -> {path}")

    return best_alpha, best_beta


# ---------------------------------------------------------------------------
# Task 5 — Beam + 4-gram LM
# ---------------------------------------------------------------------------

def task5(decoder, samples, logits_list, best_alpha, best_beta):
    print("\n" + "="*60)
    print("TASK 5 — Beam + 4-gram LM (LibriSpeech)")
    print("="*60)

    if not os.path.exists(LM_4GRAM):
        print(f"  WARNING: {LM_4GRAM} not found — skipping Task 5.")
        print("  Download from http://www.openslr.org/11/ and place at lm/4-gram.arpa.gz")
        return None

    decoder.lm_model = get_lm(LM_4GRAM)
    decoder.beam_width = 10
    decoder.alpha = best_alpha
    decoder.beta = best_beta

    m, hyps = evaluate_cached(decoder, samples, logits_list, method="beam_lm")
    print_metrics(f"4-gram (alpha={best_alpha}, beta={best_beta})", m)

    rows = [{"path": p, "ref": r, "hyp": h}
            for (p, r), h in zip(samples, hyps)]
    save_csv(f"{RESULTS_DIR}/task5_4gram.csv", rows)
    save_csv(f"{RESULTS_DIR}/task5_metrics.csv",
             [{"lm": "4-gram", "alpha": best_alpha, "beta": best_beta,
               "wer": m["wer"], "cer": m["cer"]}])
    return m


# ---------------------------------------------------------------------------
# Task 6 — LM rescoring, alpha/beta sweep + qualitative comparison
# ---------------------------------------------------------------------------

def task6(decoder, samples, logits_list):
    print("\n" + "="*60)
    print("TASK 6 — LM rescoring (LibriSpeech)")
    print("="*60)

    decoder.lm_model = get_lm(LM_3GRAM)
    decoder.beam_width = 10

    alphas = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    betas  = [0.0, 0.5, 1.0, 1.5]

    rows = []
    best_wer = float("inf")
    best_alpha, best_beta = 0.1, 0.5

    for alpha in alphas:
        for beta in betas:
            print(f"  alpha={alpha}, beta={beta} ...", flush=True)
            decoder.alpha = alpha
            decoder.beta = beta
            m, _ = evaluate_cached(decoder, samples, logits_list,
                                   method="beam_lm_rescore")
            print_metrics(f"    alpha={alpha} beta={beta}", m)
            rows.append({"alpha": alpha, "beta": beta,
                         "wer": m["wer"], "cer": m["cer"]})
            if m["wer"] < best_wer:
                best_wer = m["wer"]
                best_alpha, best_beta = alpha, beta

    save_csv(f"{RESULTS_DIR}/task6_rescore_sweep.csv", rows)
    print(f"\n  Best rescoring config: alpha={best_alpha}, beta={best_beta}, WER={best_wer:.2%}")

    # Qualitative comparison
    print("\n  --- Qualitative comparison (beam vs SF vs rescoring) ---")
    decoder.alpha = best_alpha
    decoder.beta = best_beta

    qual_rows = []
    found = 0
    for (path, ref), logits in zip(samples, logits_list):
        if found >= 10:
            break
        hyp_beam = decode_logits(decoder, logits, "beam")
        hyp_sf   = decode_logits(decoder, logits, "beam_lm")
        hyp_rs   = decode_logits(decoder, logits, "beam_lm_rescore")
        if hyp_sf != hyp_beam or hyp_rs != hyp_beam:
            found += 1
            qual_rows.append({
                "path": path, "ref": ref,
                "beam": hyp_beam, "sf": hyp_sf, "rescore": hyp_rs
            })
            sf_mark = "corrected" if hyp_sf == ref else ("error" if hyp_sf != hyp_beam else "same")
            rs_mark = "corrected" if hyp_rs == ref else ("error" if hyp_rs != hyp_beam else "same")
            print(f"\n  REF : {ref}")
            print(f"  BEAM: {hyp_beam}")
            print(f"  SF  : {hyp_sf}   [{sf_mark}]")
            print(f"  RS  : {hyp_rs}   [{rs_mark}]")

    save_csv(f"{RESULTS_DIR}/task6_qualitative.csv", qual_rows)
    return best_alpha, best_beta


# ---------------------------------------------------------------------------
# Task 7 — All methods on both test sets + temperature sweep with LM (7b)
# ---------------------------------------------------------------------------

def task7(decoder, libri_samples, earnings_samples,
          libri_logits, earnings_logits,
          best_alpha_sf, best_beta_sf,
          best_alpha_rs, best_beta_rs):
    print("\n" + "="*60)
    print("TASK 7 — All methods on both test sets (domain comparison)")
    print("="*60)

    configs = [
        ("greedy",          None,      1.0,           0.0,          1),
        ("beam",            None,      1.0,           0.0,          10),
        ("beam_lm",         LM_3GRAM,  best_alpha_sf, best_beta_sf, 10),
        ("beam_lm_rescore", LM_3GRAM,  best_alpha_rs, best_beta_rs, 10),
    ]
    labels = ["Greedy", "Beam search", "Beam + 3-gram (SF)", "Beam + 3-gram (RS)"]

    comparison_rows = []
    for (method, lm, alpha, beta, bw), label in zip(configs, labels):
        print(f"\n  [{label}]")
        decoder.lm_model = get_lm(lm)
        decoder.beam_width = bw
        decoder.alpha = alpha
        decoder.beta = beta
        m_lib,  _ = evaluate_cached(decoder, libri_samples, libri_logits, method=method)
        m_earn, _ = evaluate_cached(decoder, earnings_samples, earnings_logits, method=method)
        print_metrics("  LibriSpeech", m_lib)
        print_metrics("  Earnings22 ", m_earn)
        comparison_rows.append({
            "method": label,
            "libri_wer": m_lib["wer"],  "libri_cer": m_lib["cer"],
            "earn_wer":  m_earn["wer"], "earn_cer":  m_earn["cer"],
        })

    save_csv(f"{RESULTS_DIR}/task7_comparison.csv", comparison_rows)

    print("\n  Summary table:")
    print(f"  {'Method':<30} {'LibriSpeech WER':>16} {'LibriSpeech CER':>16} "
          f"{'Earnings22 WER':>15} {'Earnings22 CER':>15}")
    for r in comparison_rows:
        print(f"  {r['method']:<30} {r['libri_wer']:>15.2%} {r['libri_cer']:>16.2%} "
              f"{r['earn_wer']:>14.2%} {r['earn_cer']:>15.2%}")

    # Task 7b — Temperature sweep on Earnings22
    print("\n" + "-"*60)
    print("TASK 7b — Temperature sweep on Earnings22 (greedy + SF)")
    print("-"*60)

    decoder.lm_model = get_lm(LM_3GRAM)
    decoder.beam_width = 10
    decoder.alpha = best_alpha_sf
    decoder.beta = best_beta_sf

    temps = [0.5, 1.0, 1.5, 2.0]
    temp_rows = []
    for T in temps:
        print(f"  T={T} ...", flush=True)
        m_g,  _ = evaluate_cached(decoder, earnings_samples, earnings_logits,
                                  method="greedy", temperature=T)
        m_sf, _ = evaluate_cached(decoder, earnings_samples, earnings_logits,
                                  method="beam_lm", temperature=T)
        print_metrics(f"  T={T} greedy", m_g)
        print_metrics(f"  T={T} SF    ", m_sf)
        temp_rows.append({
            "temperature": T,
            "greedy_wer": m_g["wer"],  "greedy_cer": m_g["cer"],
            "sf_wer":     m_sf["wer"], "sf_cer":     m_sf["cer"],
        })

    save_csv(f"{RESULTS_DIR}/task7b_temperature_sweep.csv", temp_rows)

    df = pd.DataFrame(temp_rows)
    fig, ax = plt.subplots()
    ax.plot(df["temperature"], df["greedy_wer"] * 100, marker="o", label="Greedy")
    ax.plot(df["temperature"], df["sf_wer"] * 100,     marker="s", label="Beam+LM (SF)")
    ax.set_xlabel("Temperature")
    ax.set_ylabel("WER (%)")
    ax.set_title("Task 7b - Temperature vs WER (Earnings22)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    path = f"{RESULTS_DIR}/task7b_temperature_sweep.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved -> {path}")


# ---------------------------------------------------------------------------
# Task 8 — Train financial KenLM
# ---------------------------------------------------------------------------

def task8():
    print("\n" + "="*60)
    print("TASK 8 — Train financial-domain KenLM")
    print("="*60)

    corpus = "data/earnings22_train/corpus.txt"
    arpa   = "/tmp/financial-3gram.arpa"
    out_gz = LM_FINANCIAL

    if os.path.exists(out_gz):
        print(f"  Financial LM already exists: {out_gz}")
        return

    lmplz_bin = "/tmp/kenlm_build/build/bin/lmplz"

    if not os.path.exists(lmplz_bin):
        print("  KenLM binaries not found. Build them first with:")
        print()
        print("    sudo apt-get install cmake libboost-all-dev")
        print("    git clone --depth=1 https://github.com/kpu/kenlm /tmp/kenlm_build")
        print("    mkdir /tmp/kenlm_build/build && cd /tmp/kenlm_build/build")
        print("    cmake .. && make -j4 lmplz build_binary")
        print()
        print("  Then re-run: python run_experiments.py --task 8")
        return

    import subprocess, gzip, shutil
    print(f"  Training 3-gram LM from {corpus} ...")
    with open(arpa, "w") as f_out:
        ret = subprocess.run(
            [lmplz_bin, "-o", "3", "--discount_fallback"],
            stdin=open(corpus, "r"), stdout=f_out, stderr=subprocess.PIPE
        )
    if ret.returncode != 0:
        print("  ERROR during lmplz:", ret.stderr.decode())
        return

    os.makedirs("lm", exist_ok=True)
    with open(arpa, "rb") as f_in, gzip.open(out_gz, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    print(f"  Financial LM saved -> {out_gz}")


# ---------------------------------------------------------------------------
# Task 9 — All LMs on both test sets, bar chart
# ---------------------------------------------------------------------------

def task9(decoder, libri_samples, earnings_samples,
          libri_logits, earnings_logits,
          best_alpha_sf, best_beta_sf,
          best_alpha_rs, best_beta_rs):
    print("\n" + "="*60)
    print("TASK 9 — All LMs on both test sets")
    print("="*60)

    lm_configs = [("3-gram LibriSpeech", LM_3GRAM)]
    if os.path.exists(LM_FINANCIAL):
        lm_configs.append(("financial 3-gram", LM_FINANCIAL))
    if os.path.exists(LM_4GRAM):
        lm_configs.append(("4-gram LibriSpeech", LM_4GRAM))

    decoder.beam_width = 10
    rows = []
    for lm_name, lm_path in lm_configs:
        print(f"\n  LM: {lm_name}")
        decoder.lm_model = get_lm(lm_path)
        for method, alpha, beta, label in [
            ("beam_lm",         best_alpha_sf, best_beta_sf, "Shallow Fusion"),
            ("beam_lm_rescore", best_alpha_rs, best_beta_rs, "Rescoring"),
        ]:
            decoder.alpha = alpha
            decoder.beta = beta
            m_lib,  _ = evaluate_cached(decoder, libri_samples, libri_logits, method=method)
            m_earn, _ = evaluate_cached(decoder, earnings_samples, earnings_logits, method=method)
            key = f"{lm_name} ({label})"
            print_metrics(f"  {key} LibriSpeech", m_lib)
            print_metrics(f"  {key} Earnings22 ", m_earn)
            rows.append({
                "lm": lm_name, "method": label,
                "libri_wer": m_lib["wer"], "libri_cer": m_lib["cer"],
                "earn_wer":  m_earn["wer"], "earn_cer":  m_earn["cer"],
            })

    if not rows:
        print("  No LMs available for Task 9.")
        return

    save_csv(f"{RESULTS_DIR}/task9_all_lms.csv", rows)

    # Bar chart
    df = pd.DataFrame(rows)
    x_labels = [f"{r['lm']}\n({r['method']})" for _, r in df.iterrows()]
    x = list(range(len(df)))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(df) * 2), 5))
    ax.bar([i - width/2 for i in x], df["libri_wer"] * 100,
           width, label="LibriSpeech WER", color="steelblue")
    ax.bar([i + width/2 for i in x], df["earn_wer"] * 100,
           width, label="Earnings22 WER", color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_ylabel("WER (%)")
    ax.set_title("Task 9 - WER by LM and Method")
    ax.legend()
    ax.grid(axis="y")
    fig.tight_layout()
    path = f"{RESULTS_DIR}/task9_bar_chart.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="ASR Decoding Assignment 2 — full experiment runner"
    )
    parser.add_argument("--task", type=int, default=None,
                        help="Run only task N (1-9). Omit to run all tasks.")
    parser.add_argument("--quick", action="store_true",
                        help="Use only first 20 samples per dataset (for debugging).")
    return parser.parse_args()


def main():
    args = parse_args()
    limit = 20 if args.quick else None
    ensure_dir(RESULTS_DIR)

    print("Loading datasets ...")
    libri_samples    = load_manifest(LIBRI_MANIFEST,    limit=limit)
    earnings_samples = load_manifest(EARNINGS_MANIFEST, limit=limit)
    print(f"  LibriSpeech test-other: {len(libri_samples)} samples")
    print(f"  Earnings22 test:        {len(earnings_samples)} samples")

    # Load acoustic model ONCE
    print("Loading acoustic model ...")
    decoder = Wav2Vec2Decoder(lm_model_path=None)

    # Pre-compute logits for all samples (model inference done once)
    print("Pre-computing logits for LibriSpeech ...")
    libri_logits = precompute_logits(decoder, libri_samples)
    print("Pre-computing logits for Earnings22 ...")
    earnings_logits = precompute_logits(decoder, earnings_samples)

    run_all = args.task is None

    # Best hyperparams — updated by Task 4 and 6 if they run
    best_alpha_sf, best_beta_sf = 0.1, 0.5   # shallow fusion defaults
    best_alpha_rs, best_beta_rs = 0.1, 0.5   # rescoring defaults

    if run_all or args.task == 1:
        task1(decoder, libri_samples, libri_logits)

    if run_all or args.task == 2:
        task2(decoder, libri_samples, libri_logits)

    if run_all or args.task == 3:
        task3(decoder, libri_samples, libri_logits)

    if run_all or args.task == 4:
        result = task4(decoder, libri_samples, libri_logits)
        if result:
            best_alpha_sf, best_beta_sf = result

    if run_all or args.task == 5:
        task5(decoder, libri_samples, libri_logits, best_alpha_sf, best_beta_sf)

    if run_all or args.task == 6:
        result = task6(decoder, libri_samples, libri_logits)
        if result:
            best_alpha_rs, best_beta_rs = result

    if run_all or args.task == 7:
        task7(decoder, libri_samples, earnings_samples,
              libri_logits, earnings_logits,
              best_alpha_sf, best_beta_sf,
              best_alpha_rs, best_beta_rs)

    if run_all or args.task == 8:
        task8()

    if run_all or args.task == 9:
        task9(decoder, libri_samples, earnings_samples,
              libri_logits, earnings_logits,
              best_alpha_sf, best_beta_sf,
              best_alpha_rs, best_beta_rs)

    print("\n" + "="*60)
    print(f"Done! All results saved in '{RESULTS_DIR}/'")
    print("="*60)


if __name__ == "__main__":
    main()
