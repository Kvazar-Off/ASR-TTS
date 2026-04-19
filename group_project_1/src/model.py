"""
Small Conformer-CTC model for Russian number ASR.
Target: ≤ 5M parameters.

Architecture:
  MelSpec (80) → ConvSubsampling → Linear → 6×ConformerBlock → Linear → vocab
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Positional Encoding                                                        #
# --------------------------------------------------------------------------- #

class RelativePositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (absolute, added to input)."""

    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, T, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# --------------------------------------------------------------------------- #
#  Conformer sub-modules                                                      #
# --------------------------------------------------------------------------- #

class FeedForwardModule(nn.Module):
    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * expansion),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expansion, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiHeadSelfAttentionModule(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        return self.dropout(x)


class ConvolutionModule(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size must be odd"
        padding = (kernel_size - 1) // 2
        self.norm = nn.LayerNorm(d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),        # pointwise expand
            nn.GLU(dim=-1),                           # → d_model
            # depthwise conv
            nn.Conv1d(d_model, d_model, kernel_size, padding=padding, groups=d_model),
            nn.BatchNorm1d(d_model),
            nn.SiLU(),
            nn.Conv1d(d_model, d_model, 1),           # pointwise project
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        residual = x
        x = self.norm(x)
        # conv expects (B, C, T)
        B, T, C = x.shape
        out = x
        # split linear + GLU inline
        out = nn.functional.linear(out, self.net[0].weight, self.net[0].bias)
        out = torch.sigmoid(out[..., C:]) * out[..., :C]  # GLU
        out = out.transpose(1, 2)                         # (B, C, T)
        out = self.net[2](out)                            # depthwise conv
        out = self.net[3](out)                            # BN
        out = self.net[4](out)                            # SiLU
        out = self.net[5](out)                            # pointwise
        out = self.net[6](out)                            # dropout
        out = out.transpose(1, 2)                         # (B, T, C)
        return out


class ConformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int = 144,
        num_heads: int = 4,
        ff_expansion: int = 4,
        conv_kernel: int = 31,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ff1 = FeedForwardModule(d_model, ff_expansion, dropout)
        self.attn = MultiHeadSelfAttentionModule(d_model, num_heads, dropout)
        self.conv = ConvolutionModule(d_model, conv_kernel, dropout)
        self.ff2 = FeedForwardModule(d_model, ff_expansion, dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + 0.5 * self.ff1(x)
        x = x + self.attn(x, key_padding_mask)
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        return self.norm(x)


# --------------------------------------------------------------------------- #
#  Conv subsampling  (factor 4×)                                              #
# --------------------------------------------------------------------------- #

class ConvSubsampling(nn.Module):
    """
    2× stride conv applied twice → 4× time reduction.
    Input: (B, 1, T, n_mels)  →  Output: (B, d_model, T//4, 1)
    """

    def __init__(self, n_mels: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        # compute output mel dim after 2× stride on freq axis
        out_freq = math.ceil(math.ceil(n_mels / 2) / 2)
        self.proj = nn.Linear(d_model * out_freq, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, T, n_mels)
        returns: (B, T', d_model), output_lengths (B,)
        """
        B, T, F = x.shape
        x = x.unsqueeze(1)                      # (B, 1, T, F)
        x = self.net(x)                          # (B, d_model, T', F')
        B2, C, T2, F2 = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()  # (B, T', d_model, F')
        x = x.view(B2, T2, C * F2)              # (B, T', d_model*F')
        x = self.proj(x)                         # (B, T', d_model)
        return self.dropout(x), T2


# --------------------------------------------------------------------------- #
#  Full Conformer-CTC model                                                   #
# --------------------------------------------------------------------------- #

class ConformerCTC(nn.Module):
    """
    Conformer encoder with CTC head.

    Default config (~2.5M params):
      n_mels=80, d_model=144, n_layers=6, n_heads=4, ff_expansion=4
    """

    def __init__(
        self,
        vocab_size: int,
        n_mels: int = 80,
        d_model: int = 144,
        n_layers: int = 6,
        n_heads: int = 4,
        ff_expansion: int = 4,
        conv_kernel: int = 31,
        dropout: float = 0.1,
        max_len: int = 2048,
    ):
        super().__init__()
        self.subsampling = ConvSubsampling(n_mels, d_model, dropout)
        self.pos_enc = RelativePositionalEncoding(d_model, max_len, dropout)
        self.encoder = nn.ModuleList(
            [
                ConformerBlock(d_model, n_heads, ff_expansion, conv_kernel, dropout)
                for _ in range(n_layers)
            ]
        )
        self.head = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(
        self,
        x: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, n_mels) — mel spectrogram
            input_lengths: (B,) — original frame counts before subsampling

        Returns:
            log_probs: (T', B, vocab_size)  — for CTC loss
            output_lengths: (B,)
        """
        x, T_out = self.subsampling(x)          # (B, T', d)
        output_lengths = self._compute_out_lengths(input_lengths, T_out)


        x = self.pos_enc(x)

        # Build padding mask: True = padded position (to ignore in attention)
        B, T2, _ = x.shape
        key_padding_mask = torch.arange(T2, device=x.device).unsqueeze(0) >= output_lengths.unsqueeze(1)

        for layer in self.encoder:
            x = layer(x, key_padding_mask)

        logits = self.head(x)                    # (B, T', vocab)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs.permute(1, 0, 2), output_lengths  # CTC wants (T, B, C)

    @staticmethod
    def _compute_out_lengths(lengths: torch.Tensor, T_out: int) -> torch.Tensor:
        """Approximate output lengths after 2× subsampling twice."""
        # each stride-2 conv: L_out = ceil(L_in / 2)
        l = torch.ceil(lengths.float() / 2).long()
        l = torch.ceil(l.float() / 2).long()
        return l.clamp(max=T_out)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- #
#  Sanity check                                                               #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    vocab_size = 35   # approximate
    model = ConformerCTC(vocab_size=vocab_size)
    n_params = model.count_parameters()
    print(f"Parameters: {n_params:,}  ({n_params/1e6:.2f}M)")

    # forward pass test
    B, T, F = 2, 400, 80
    x = torch.randn(B, T, F)
    lengths = torch.tensor([400, 350])
    log_probs, out_lengths = model(x, lengths)
    print(f"Input:  {x.shape}")
    print(f"Output: {log_probs.shape}  (T', B, vocab)")
    print(f"Out lengths: {out_lengths}")
