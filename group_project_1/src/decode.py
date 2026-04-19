"""
CTC decoding: greedy and simple prefix beam search.
"""
from __future__ import annotations
import torch
from collections import defaultdict
from text_utils import decode_indices, words_to_num, get_vocab


# --------------------------------------------------------------------------- #
#  Greedy decoder                                                             #
# --------------------------------------------------------------------------- #

def ctc_greedy_decode(log_probs: torch.Tensor, output_lengths: torch.Tensor) -> list[list[int]]:
    """
    log_probs:      (T, B, vocab)
    output_lengths: (B,)
    returns: list of token id sequences (blank collapsed)
    """
    T, B, V = log_probs.shape
    preds = log_probs.argmax(dim=-1)   # (T, B)
    preds = preds.T                    # (B, T)

    results = []
    for b in range(B):
        seq = preds[b, : output_lengths[b]].tolist()
        # collapse repeats + remove blank (0)
        prev = -1
        tokens = []
        for t in seq:
            if t != prev:
                if t != 0:
                    tokens.append(t)
                prev = t
        results.append(tokens)
    return results


# --------------------------------------------------------------------------- #
#  Beam search decoder (no LM)                                                #
# --------------------------------------------------------------------------- #

def ctc_beam_decode(
    log_probs: torch.Tensor,
    output_length: int,
    beam_width: int = 10,
    blank_idx: int = 0,
) -> list[int]:
    """
    Single-sample beam search CTC decoder.
    log_probs: (T, vocab)
    returns best token sequence
    """
    # prefix beam: {prefix_tuple: (prob_blank, prob_non_blank)}
    NEG_INF = float("-inf")

    beams: dict[tuple, list[float]] = {(): [0.0, NEG_INF]}   # prefix → [Pb, Pnb]

    for t in range(output_length):
        new_beams: dict[tuple, list[float]] = defaultdict(lambda: [NEG_INF, NEG_INF])

        # Get top-k for speed
        probs_t = log_probs[t]
        top_k = min(beam_width * 2, probs_t.shape[0])
        topk_vals, topk_idx = probs_t.topk(top_k)

        for prefix, (Pb, Pnb) in beams.items():
            P_total = _logadd(Pb, Pnb)

            for k in range(top_k):
                c = topk_idx[k].item()
                lp = topk_vals[k].item()

                if c == blank_idx:
                    nb = new_beams[prefix]
                    nb[0] = _logadd(nb[0], P_total + lp)
                else:
                    new_prefix = prefix + (c,)
                    # if c == last char → only blank can extend without repeat
                    if prefix and prefix[-1] == c:
                        nb = new_beams[new_prefix]
                        nb[1] = _logadd(nb[1], Pb + lp)
                        # extend with same char: only via blank predecessor
                    else:
                        nb = new_beams[new_prefix]
                        nb[1] = _logadd(nb[1], P_total + lp)

                    # handle repeat: end in same char, came from blank
                    if prefix and prefix[-1] == c:
                        nb2 = new_beams[prefix]
                        nb2[1] = _logadd(nb2[1], Pnb + lp)

        # Prune to beam_width
        beams = dict(
            sorted(
                new_beams.items(),
                key=lambda kv: _logadd(kv[1][0], kv[1][1]),
                reverse=True,
            )[:beam_width]
        )

    best = max(beams, key=lambda p: _logadd(beams[p][0], beams[p][1]))
    return list(best)


def _logadd(a: float, b: float) -> float:
    if a == float("-inf"):
        return b
    if b == float("-inf"):
        return a
    if a > b:
        return a + torch.log1p(torch.exp(torch.tensor(b - a))).item()
    return b + torch.log1p(torch.exp(torch.tensor(a - b))).item()


# --------------------------------------------------------------------------- #
#  Full decode pipeline: log_probs → integer prediction                       #
# --------------------------------------------------------------------------- #

def decode_batch(
    log_probs: torch.Tensor,
    output_lengths: torch.Tensor,
    idx2char: dict[int, str],
    use_beam: bool = False,
    beam_width: int = 5,
) -> list[int | None]:
    """
    Decode a batch of log_probs to integer number predictions.

    log_probs:      (T, B, vocab)
    output_lengths: (B,)
    returns: list of int (or None if parse fails)
    """
    if use_beam:
        results = []
        T, B, V = log_probs.shape
        for b in range(B):
            tokens = ctc_beam_decode(
                log_probs[:, b, :],
                output_lengths[b].item(),
                beam_width=beam_width,
            )
            results.append(tokens)
    else:
        results = ctc_greedy_decode(log_probs, output_lengths)

    preds = []
    for tokens in results:
        text = decode_indices(tokens, idx2char)
        text = text.strip()
        num = words_to_num(text)
        # clamp to valid range
        if num is not None:
            num = max(1000, min(999999, num))
        preds.append(num)
    return preds


# --------------------------------------------------------------------------- #
#  CER computation                                                            #
# --------------------------------------------------------------------------- #

def char_error_rate(reference: str, hypothesis: str) -> float:
    """Character Error Rate (edit distance / len(ref))."""
    r, h = list(reference), list(hypothesis)
    n, m = len(r), len(h)
    if n == 0:
        return float(m > 0)
    # DP
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): dp[i][0] = i
    for j in range(m + 1): dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if r[i - 1] == h[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[n][m] / n


def compute_cer_batch(references: list[str], hypotheses: list[str]) -> float:
    """Mean CER over a batch."""
    if not references:
        return 0.0
    return sum(char_error_rate(r, h) for r, h in zip(references, hypotheses)) / len(references)
