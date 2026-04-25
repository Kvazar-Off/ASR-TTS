"""
PyTorch Dataset for Russian number ASR competition.

Handles:
  - resample to 16 kHz
  - wav + mp3 formats
  - text normalization: int → words → char indices
  - per-utterance CMVN (default) or global CMVN
  - waveform augmentations: speed, pitch shift, MP3 codec, RIR reverb, noise
  - mel-level augmentations: VTLP, SpecAugment, time warp
"""
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as AF
import torchaudio.transforms as T
from torch.utils.data import Dataset

from text_utils import get_vocab, encode_text, num_to_words

try:
    from torchaudio.io import AudioEffector
    _HAS_EFFECTOR = True
except Exception:
    AudioEffector = None
    _HAS_EFFECTOR = False


def load_audio(path: str | Path) -> tuple[torch.Tensor, int]:
    """Load audio via soundfile (avoids FFmpeg/torchcodec dependency on Windows)."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    # soundfile returns (samples, channels); torchaudio convention is (channels, samples)
    waveform = torch.from_numpy(np.ascontiguousarray(data.T))
    return waveform, int(sr)


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

# Cache T.Resample modules keyed by (orig_sr, target_sr).
# Constructing a Resample module recomputes a windowed-sinc kernel — doing it
# inline per-sample (as the original code did) was the dominant CPU cost.
_resample_cache: dict[tuple[int, int], T.Resample] = {}


def _freeze(module: torch.nn.Module) -> torch.nn.Module:
    module.eval()
    for p in module.parameters():
        try:
            p.requires_grad_(False)
        except Exception:
            # Skip uninitialized lazy parameters; they will be frozen via no_grad context.
            pass
    for b in module.buffers():
        try:
            b.requires_grad_(False)
        except Exception:
            pass
    return module


def _get_resampler(orig_sr: int, target_sr: int) -> T.Resample:
    key = (int(orig_sr), int(target_sr))
    rs = _resample_cache.get(key)
    if rs is None:
        rs = _freeze(T.Resample(key[0], key[1]))
        _resample_cache[key] = rs
    return rs


# Cache PitchShift modules per n_steps. Each module precomputes STFT window/buffers.
_pitch_shift_cache: dict[int, "T.PitchShift"] = {}


def _get_pitch_shifter(n_steps: int) -> "T.PitchShift":
    n_steps = int(n_steps)
    ps = _pitch_shift_cache.get(n_steps)
    if ps is None:
        ps = _freeze(T.PitchShift(TARGET_SR, n_steps=n_steps))
        _pitch_shift_cache[n_steps] = ps
    return ps


_freeze(_mel_transform)


def _to_mono_16k(waveform: torch.Tensor, sr: int) -> torch.Tensor:
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True)
    if sr != TARGET_SR:
        waveform = _get_resampler(sr, TARGET_SR)(waveform)
    return waveform


def wav_to_mel(waveform: torch.Tensor, sr: int) -> torch.Tensor:
    """
    waveform: (channels, T)  or  (T,)
    returns:  (T_frames, N_MELS)  — log mel spectrogram
    """
    waveform = _to_mono_16k(waveform, sr)

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
        csv_path:
        data_root
        augment:    apply waveform + mel augmentations
        noise_dir:  optional folder with .wav noise files for SNR mixing
        rir_dir:    optional folder with RIR .wav files for reverb augmentation
        cmvn_mode:  "utterance" (per-file CMVN, default) or "global" (uses mel_mean/std)
    """

    def __init__(
        self,
        csv_path: str | Path,
        data_root: str | Path,
        augment: bool = False,
        noise_dir: Optional[str | Path] = None,
        rir_dir: Optional[str | Path] = None,
        mel_mean: Optional[torch.Tensor] = None,
        mel_std: Optional[torch.Tensor] = None,
        cmvn_mode: str = "utterance",
        max_aug_clips: int = 500,
        use_pitch_shift: bool = True,
    ):
        if cmvn_mode not in {"utterance", "global"}:
            raise ValueError(f"Unknown cmvn_mode: {cmvn_mode}")

        self.df = pd.read_csv(csv_path)
        self.data_root = Path(data_root)
        self.augment = augment
        self.noise_clips: list[torch.Tensor] = []
        self.rir_clips: list[torch.Tensor] = []
        self.mel_mean = mel_mean
        self.mel_std = mel_std
        self.cmvn_mode = cmvn_mode
        self.max_aug_clips = max_aug_clips
        self.use_pitch_shift = use_pitch_shift

        if noise_dir is not None:
            self._load_noise(Path(noise_dir))
        if rir_dir is not None:
            self._load_rirs(Path(rir_dir))

        self.char2idx, self.idx2char = get_vocab()

    def _load_noise(self, noise_dir: Path):
        paths = list(noise_dir.glob("*.wav"))
        if len(paths) > self.max_aug_clips:
            random.Random(0).shuffle(paths)
            paths = paths[: self.max_aug_clips]
        for p in paths:
            try:
                w, sr = load_audio(p)
                if sr != TARGET_SR:
                    w = _get_resampler(sr, TARGET_SR)(w)
                self.noise_clips.append(w.mean(0))
            except Exception:
                pass
        print(f"Loaded {len(self.noise_clips)} noise clips from {noise_dir}")

    def _load_rirs(self, rir_dir: Path):
        paths = list(rir_dir.glob("**/*.wav"))
        if len(paths) > self.max_aug_clips:
            random.Random(0).shuffle(paths)
            paths = paths[: self.max_aug_clips]
        for p in paths:
            try:
                w, sr = load_audio(p)
                if sr != TARGET_SR:
                    w = _get_resampler(sr, TARGET_SR)(w)
                w = w.mean(0)
                # Trim leading silence + cap length to 0.5s for speed
                peak = w.abs().argmax().item()
                start = max(0, peak - int(0.005 * TARGET_SR))
                w = w[start:start + int(0.5 * TARGET_SR)]
                if w.numel() < 16:
                    continue
                w = w / w.abs().max().clamp(min=1e-6)
                self.rir_clips.append(w)
            except Exception:
                pass
        print(f"Loaded {len(self.rir_clips)} RIR clips from {rir_dir}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        with torch.no_grad():
            return self._get_item(idx)

    def _get_item(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        audio_path = self.data_root / row["filename"]

        waveform, sr = load_audio(audio_path)
        waveform = _to_mono_16k(waveform, sr)
        sr = TARGET_SR

        if self.augment:
            waveform = self._apply_waveform_augmentations(waveform)

        if self.augment and self.rir_clips and random.random() < 0.3:
            waveform = self._add_reverb(waveform)

        if self.augment and self.noise_clips and random.random() < 0.4:
            waveform = self._add_noise(waveform, sr)

        mel = wav_to_mel(waveform, sr) # (T, n_mels)

        mel = self._normalize_mel(mel)

        if self.augment and random.random() < 0.4:
            mel = self._vtlp(mel)

        if self.augment:
            mel = spec_augment(mel)

        # Time-warp: stretch/squeeze by 5%
        if self.augment and random.random() < 0.3:
            mel = self._time_warp(mel)

        # Encode label
        label_int = int(row["transcription"])
        label_words = num_to_words(label_int)
        label_ids = encode_text(label_words, self.char2idx)

        return {
            "mel": mel, # (T, n_mels)
            "label": torch.tensor(label_ids, dtype=torch.long),
            "mel_len": mel.shape[0],
            "label_len": len(label_ids),
            "filename": row["filename"],
            "transcription": label_int,
            "spk_id": row["spk_id"] if "spk_id" in row else None,
            "gender": row["gender"] if "gender" in row else None,
            "ext": row["ext"] if "ext" in row else Path(row["filename"]).suffix.lstrip("."),
        }

    def _apply_waveform_augmentations(self, waveform: torch.Tensor) -> torch.Tensor:
        waveform = waveform.clone()

        # Speed perturbation (changes duration; safe for word-level CTC targets)
        if random.random() < 0.5:
            factor = random.choice([0.9, 0.95, 1.05, 1.1])
            waveform = self._speed_perturb(waveform, factor)

        # Pitch shift (simulates new speakers without changing duration)
        if self.use_pitch_shift and random.random() < 0.3:
            n_steps = random.choice([-2, -1, 1, 2])
            waveform = self._pitch_shift(waveform, n_steps)

        if random.random() < 0.5:
            gain_db = random.uniform(-6.0, 6.0)
            waveform = waveform * (10 ** (gain_db / 20.0))

        # Real MP3 codec round-trip (preferred) or bitcrush fallback
        if random.random() < 0.35:
            waveform = self._codec_degradation(waveform)

        if random.random() < 0.25:
            waveform = self._bandlimit_degradation(waveform)

        if random.random() < 0.15:
            waveform = self._temporal_dropout(waveform)

        if random.random() < 0.2:
            waveform = torch.tanh(waveform * random.uniform(1.05, 1.5))

        return waveform.clamp(min=-1.0, max=1.0)

    @staticmethod
    def _speed_perturb(waveform: torch.Tensor, factor: float) -> torch.Tensor:
        if abs(factor - 1.0) < 1e-3:
            return waveform
        try:
            out, _ = AF.speed(waveform, TARGET_SR, factor)
            return out
        except Exception:
            new_len = max(1, int(waveform.shape[-1] / factor))
            return F.interpolate(
                waveform.unsqueeze(0), size=new_len, mode="linear", align_corners=False
            ).squeeze(0)

    @staticmethod
    def _pitch_shift(waveform: torch.Tensor, n_steps: int) -> torch.Tensor:
        if n_steps == 0:
            return waveform
        try:
            shifter = _get_pitch_shifter(int(n_steps))
            return shifter(waveform)
        except Exception:
            return waveform

    def _codec_degradation(self, waveform: torch.Tensor) -> torch.Tensor:
        """Prefer real MP3/AAC round-trip; fall back to bitcrush+lowpass."""
        if _HAS_EFFECTOR and random.random() < 0.7:
            out = self._real_codec_roundtrip(waveform)
            if out is not None:
                return out
        return self._codec_like_degradation(waveform)

    @staticmethod
    def _real_codec_roundtrip(waveform: torch.Tensor) -> Optional[torch.Tensor]:
        """Encode/decode via ffmpeg-backed AudioEffector. Returns None on failure."""
        try:
            fmt, codec, br = random.choice([
                ("mp3", None, 24_000),
                ("mp3", None, 32_000),
                ("mp3", None, 64_000),
                ("ogg", "libvorbis", 48_000),
                ("g722", None, 64_000),
            ])
            effector = AudioEffector(format=fmt, encoder=codec)
            wav = waveform
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
            wav = wav.transpose(0, 1).contiguous()  # (T, C) for AudioEffector
            out = effector.apply(wav, TARGET_SR)
            out = out.transpose(0, 1)  # (C, T)
            if out.shape[0] > 1:
                out = out.mean(0, keepdim=True)
            # Length may differ slightly: trim/pad to original
            target_len = waveform.shape[-1]
            cur_len = out.shape[-1]
            if cur_len > target_len:
                out = out[..., :target_len]
            elif cur_len < target_len:
                pad = target_len - cur_len
                out = F.pad(out, (0, pad))
            if waveform.dim() == 1:
                out = out.squeeze(0)
            return out
        except Exception:
            return None

    @staticmethod
    def _codec_like_degradation(waveform: torch.Tensor) -> torch.Tensor:
        down_sr = random.choice([8_000, 10_000, 12_000])
        degraded = _get_resampler(TARGET_SR, down_sr)(waveform)
        degraded = _get_resampler(down_sr, TARGET_SR)(degraded)

        bits = random.choice([6, 7, 8])
        levels = float(2 ** bits - 1)
        degraded = torch.round((degraded.clamp(-1.0, 1.0) + 1.0) * 0.5 * levels) / levels
        degraded = degraded * 2.0 - 1.0

        if random.random() < 0.5:
            cutoff = random.choice([2800, 3200, 3600, 4200])
            degraded = AF.lowpass_biquad(degraded, TARGET_SR, cutoff)

        return degraded

    @staticmethod
    def _bandlimit_degradation(waveform: torch.Tensor) -> torch.Tensor:
        down_sr = random.choice([12_000, 14_000])
        degraded = _get_resampler(TARGET_SR, down_sr)(waveform)
        degraded = _get_resampler(down_sr, TARGET_SR)(degraded)
        if random.random() < 0.5:
            cutoff = random.choice([3000, 3800, 5000, 6500])
            degraded = AF.lowpass_biquad(degraded, TARGET_SR, cutoff)
        return degraded

    @staticmethod
    def _temporal_dropout(waveform: torch.Tensor) -> torch.Tensor:
        total = waveform.shape[-1]
        max_span = max(1, int(total * 0.03))
        for _ in range(random.randint(1, 3)):
            span = random.randint(max(1, max_span // 4), max_span)
            start = random.randint(0, max(0, total - span))
            waveform[..., start:start + span] *= random.uniform(0.0, 0.3)
        return waveform

    def _add_noise(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        """Mix a random noise clip at a random SNR."""
        if not self.noise_clips:
            return waveform

        waveform = _to_mono_16k(waveform, sr)

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
        mel_t = mel.T.unsqueeze(0)  # (1, F, T)
        mel_t = F.interpolate(mel_t, size=new_T, mode="linear", align_corners=False)
        return mel_t.squeeze(0).T  # (new_T, F)

    def _normalize_mel(self, mel: torch.Tensor) -> torch.Tensor:
        """Per-utterance CMVN by default, optional global stats."""
        if self.cmvn_mode == "utterance":
            mean = mel.mean(dim=0, keepdim=True)
            std = mel.std(dim=0, keepdim=True).clamp(min=1e-5)
            return (mel - mean) / std
        if self.mel_mean is not None and self.mel_std is not None:
            return (mel - self.mel_mean) / self.mel_std
        return mel

    @staticmethod
    def _vtlp(mel: torch.Tensor) -> torch.Tensor:
        """Vocal-tract-length perturbation: linear warp on mel-frequency axis."""
        T_len, F_len = mel.shape
        factor = random.uniform(0.9, 1.1)
        new_F = max(1, int(round(F_len * factor)))
        mel_t = mel.unsqueeze(0).unsqueeze(0)  # (1, 1, T, F)
        warped = F.interpolate(
            mel_t, size=(T_len, new_F), mode="bilinear", align_corners=False
        ).squeeze(0).squeeze(0)
        if new_F >= F_len:
            return warped[:, :F_len]
        pad_value = mel.min().item()
        pad = torch.full((T_len, F_len - new_F), pad_value, dtype=mel.dtype)
        return torch.cat([warped, pad], dim=1)

    def _add_reverb(self, waveform: torch.Tensor) -> torch.Tensor:
        if not self.rir_clips:
            return waveform
        rir = random.choice(self.rir_clips).to(waveform.dtype)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        signal_rms = waveform.pow(2).mean().sqrt().clamp(min=1e-6)
        kernel = rir.flip(0).unsqueeze(0).unsqueeze(0)
        padded = F.pad(waveform.unsqueeze(0), (rir.shape[-1] - 1, 0))
        out = F.conv1d(padded, kernel).squeeze(0)
        out = out[..., : waveform.shape[-1]]

        out_rms = out.pow(2).mean().sqrt().clamp(min=1e-6)
        out = out * (signal_rms / out_rms)
        return out.clamp(min=-1.0, max=1.0)


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
        "spk_id": [b.get("spk_id") for b in batch],
        "gender": [b.get("gender") for b in batch],
        "ext": [b.get("ext") for b in batch],
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
