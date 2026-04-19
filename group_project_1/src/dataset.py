"""
PyTorch Dataset for Russian number ASR competition.

Handles:
  - variable sample rates → resample to 16 kHz
  - wav + mp3 formats (via torchaudio)
  - text normalization: int → Russian words → char indices
  - SpecAugment + additive noise augmentation
"""
from __future__ import annotations
import random
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset

from text_utils import get_vocab, encode_text, num_to_words


# --------------------------------------------------------------------------- #
#  Constants                                                                  #
# --------------------------------------------------------------------------- #

TARGET_SR = 16_000
N_MELS = 80
N_FFT = 400        # 25 ms @ 16 kHz
HOP_LENGTH = 160   # 10 ms @ 16 kHz


# --------------------------------------------------------------------------- #
#  Mel spectrogram transform                    
# --------------------------------------------------------------------------- #

def make_mel_transform(sample_rate: int = TARGET_SR) -> T.MelSpectrogram:
    return T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        f_min=80.0,
        f_max=7600.0,
        norm="slaney",
        mel_scale="slaney",
    )


_mel_transform = make_mel_transform()


def wav_to_mel(waveform: torch.Tensor, sr: int) -> torch.Tensor:
    """
    waveform: (channels, T)  or  (T,)
    returns:  (T_frames, N_MELS)  — log mel spectrogram
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True)

    if sr != TARGET_SR:
        resampler = T.Resample(sr, TARGET_SR)
        waveform = resampler(waveform)

    mel = _mel_transform(waveform)          # (1, n_mels, T)
    mel = mel.squeeze(0).T                  # (T, n_mels)
    mel = torch.log(mel.clamp(min=1e-9))
    return mel


# --------------------------------------------------------------------------- #
#  SpecAugment                                                                #
# --------------------------------------------------------------------------- #

def spec_augment(
    mel: torch.Tensor,
    freq_mask_param: int = 15,
    time_mask_param: int = 35,
    num_freq_masks: int = 2,
    num_time_masks: int = 2,
) -> torch.Tensor:
    """
    mel: (T, F)
    Applies frequency and time masking in-place.
    """
    mel = mel.clone()
    T_len, F_len = mel.shape

    # Frequency masking
    for _ in range(num_freq_masks):
        f = random.randint(0, freq_mask_param)
        f0 = random.randint(0, max(0, F_len - f))
        mel[:, f0: f0 + f] = 0.0

    # Time masking
    for _ in range(num_time_masks):
        t = random.randint(0, min(time_mask_param, T_len - 1))
        t0 = random.randint(0, max(0, T_len - t))
        mel[t0: t0 + t, :] = 0.0

    return mel


# --------------------------------------------------------------------------- #
#  Dataset                                                                    #
# --------------------------------------------------------------------------- #

class NumbersDataset(Dataset):
    """
    Args:
        csv_path:   path to train.csv / dev.csv
        data_root:  root folder containing audio files
        augment:    apply SpecAugment (train only)
        noise_dir:  optional folder with .wav noise files for SNR mixing
    """

    def __init__(
        self,
        csv_path: str | Path,
        data_root: str | Path,
        augment: bool = False,
        noise_dir: Optional[str | Path] = None,
        mel_mean: Optional[torch.Tensor] = None,
        mel_std: Optional[torch.Tensor] = None,
    ):
        self.df = pd.read_csv(csv_path)
        self.data_root = Path(data_root)
        self.augment = augment
        self.noise_clips: list[torch.Tensor] = []
        self.mel_mean = mel_mean
        self.mel_std = mel_std

        if noise_dir is not None:
            self._load_noise(Path(noise_dir))

        self.char2idx, self.idx2char = get_vocab()

    def _load_noise(self, noise_dir: Path):
        for p in noise_dir.glob("*.wav"):
            try:
                w, sr = torchaudio.load(str(p))
                if sr != TARGET_SR:
                    w = T.Resample(sr, TARGET_SR)(w)
                self.noise_clips.append(w.mean(0))
            except Exception:
                pass
        print(f"Loaded {len(self.noise_clips)} noise clips from {noise_dir}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        audio_path = self.data_root / row["filename"]

        waveform, sr = torchaudio.load(str(audio_path))

        # Optional: additive noise augmentation
        if self.augment and self.noise_clips and random.random() < 0.4:
            waveform = self._add_noise(waveform, sr)

        mel = wav_to_mel(waveform, sr)   # (T, n_mels)

        if self.mel_mean is not None and self.mel_std is not None:
            mel = (mel - self.mel_mean) / self.mel_std

        if self.augment:
            mel = spec_augment(mel)

        # Time-warp: stretch/squeeze by ±5%
        if self.augment and random.random() < 0.3:
            mel = self._time_warp(mel)

        # Encode label
        label_int = int(row["transcription"])
        label_words = num_to_words(label_int)
        label_ids = encode_text(label_words, self.char2idx)

        return {
            "mel": mel,                          # (T, n_mels)
            "label": torch.tensor(label_ids, dtype=torch.long),
            "mel_len": mel.shape[0],
            "label_len": len(label_ids),
            "filename": row["filename"],
            "transcription": label_int,
        }

    def _add_noise(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        """Mix a random noise clip at a random SNR."""
        if not self.noise_clips:
            return waveform

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)
        if sr != TARGET_SR:
            waveform = T.Resample(sr, TARGET_SR)(waveform)

        noise = random.choice(self.noise_clips)
        if noise.dim() == 1:
            noise = noise.unsqueeze(0)

        target_len = waveform.shape[-1]
        noise_len = noise.shape[-1]
        if noise_len < target_len:
            repeats = (target_len + noise_len - 1) // noise_len
            noise = noise.repeat(1, repeats)
        start = random.randint(0, max(0, noise.shape[-1] - target_len))
        noise = noise[:, start:start + target_len]

        snr_db = random.uniform(5.0, 20.0)
        signal_rms = waveform.pow(2).mean().sqrt().clamp(min=1e-6)
        noise_rms = noise.pow(2).mean().sqrt().clamp(min=1e-6)
        scale = signal_rms / (10 ** (snr_db / 20.0) * noise_rms)
        mixed = waveform + scale * noise
        return mixed.clamp(min=-1.0, max=1.0)

    @staticmethod
    def _time_warp(mel: torch.Tensor) -> torch.Tensor:
        """Resample time axis by random factor in [0.95, 1.05]."""
        T_len = mel.shape[0]
        factor = random.uniform(0.95, 1.05)
        new_T = max(1, int(T_len * factor))
        # use 1D interpolation over time axis
        mel_t = mel.T.unsqueeze(0)  # (1, F, T)
        mel_t = F.interpolate(mel_t, size=new_T, mode="linear", align_corners=False)
        return mel_t.squeeze(0).T  # (new_T, F)


# --------------------------------------------------------------------------- #
#  Collate function                                                           #
# --------------------------------------------------------------------------- #

def collate_fn(batch: list[dict]) -> dict:
    """
    Pad mel spectrograms and labels to batch max length.
    """
    batch.sort(key=lambda x: x["mel_len"], reverse=True)

    mel_lens = torch.tensor([b["mel_len"] for b in batch], dtype=torch.long)
    label_lens = torch.tensor([b["label_len"] for b in batch], dtype=torch.long)

    max_mel = mel_lens[0].item()
    max_label = label_lens.max().item()
    n_mels = batch[0]["mel"].shape[1]

    mels = torch.zeros(len(batch), max_mel, n_mels)
    labels = torch.zeros(len(batch), max_label, dtype=torch.long)

    for i, b in enumerate(batch):
        m_len = b["mel_len"]
        l_len = b["label_len"]
        mels[i, :m_len] = b["mel"]
        labels[i, :l_len] = b["label"]

    return {
        "mel": mels,
        "mel_len": mel_lens,
        "label": labels,
        "label_len": label_lens,
        "filename": [b["filename"] for b in batch],
        "transcription": [b["transcription"] for b in batch],
    }


# --------------------------------------------------------------------------- #
#  Normalization stats (compute from train set)                               #
# --------------------------------------------------------------------------- #

def compute_mean_std(dataset: NumbersDataset, max_samples: int = 2000) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-feature mean and std over dataset sample."""
    indices = list(range(min(max_samples, len(dataset))))
    random.shuffle(indices)
    all_mels = []
    for i in indices[:500]:
        sample = dataset[i]
        all_mels.append(sample["mel"])
    stacked = torch.cat(all_mels, dim=0)   # (total_T, n_mels)
    mean = stacked.mean(0)
    std = stacked.std(0).clamp(min=1e-5)
    return mean, std
