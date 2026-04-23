"""
Text normalization and denormalization for Russian spoken numbers.
Range: 1_000 .. 999_999
"""
from dataclasses import dataclass
import re
from functools import lru_cache
from typing import Optional

# --------------------------------------------------------------------------- #
#  Forward: integer -> words             
# --------------------------------------------------------------------------- #

def num_to_words(n: int) -> str:
    """Convert integer to lower-case Russian words string."""
    try:
        from num2words import num2words as _n2w
    except ImportError as exc:
        raise ImportError("Install num2words: pip install num2words") from exc

    return _n2w(n, lang="ru").lower().replace("-", " ").replace("  ", " ").strip()


# --------------------------------------------------------------------------- #
#  Reverse: Russian words → integer                                           #
# --------------------------------------------------------------------------- #

_UNITS = {
    "ноль": 0,
    "нуль": 0,
    "один": 1, "одна": 1, "одного": 1,
    "два": 2, "две": 2,
    "три": 3,
    "четыре": 4,
    "пять": 5,
    "шесть": 6,
    "семь": 7,
    "восемь": 8,
    "девять": 9,
}

_TEENS = {
    "десять": 10,
    "одиннадцать": 11,
    "двенадцать": 12,
    "тринадцать": 13,
    "четырнадцать": 14,
    "пятнадцать": 15,
    "шестнадцать": 16,
    "семнадцать": 17,
    "восемнадцать": 18,
    "девятнадцать": 19,
}

_TENS = {
    "двадцать": 20,
    "тридцать": 30,
    "сорок": 40,
    "пятьдесят": 50,
    "шестьдесят": 60,
    "семьдесят": 70,
    "восемьдесят": 80,
    "девяносто": 90,
}

_HUNDREDS = {
    "сто": 100,
    "двести": 200,
    "триста": 300,
    "четыреста": 400,
    "пятьсот": 500,
    "шестьсот": 600,
    "семьсот": 700,
    "восемьсот": 800,
    "девятьсот": 900,
}

_THOUSAND = {"тысяча", "тысячи", "тысяч"}

_ALL_WORDS: dict[str, int] = {
    **_UNITS,
    **_TEENS,
    **_TENS,
    **_HUNDREDS,
}

_VALID_WORDS = set(_ALL_WORDS) | _THOUSAND


@dataclass
class ParseResult:
    value: Optional[int]
    normalized_text: str
    canonical_text: str
    repaired_tokens: list[str]
    used_fuzzy: bool = False
    inferred_thousand: bool = False


def normalize_number_text(text: str) -> str:
    text = text.lower().replace("ё", "е").strip()
    text = re.sub(r"[^а-я ]", " ", text)
    return " ".join(text.split())


def _edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr.append(min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + cost,
            ))
        prev = curr
    return prev[-1]


@lru_cache(maxsize=1024)
def _segment_compound_token(token: str) -> tuple[str, ...] | None:
    if token in _VALID_WORDS:
        return (token,)

    best: tuple[str, ...] | None = None

    def dfs(start: int, parts: list[str]) -> None:
        nonlocal best
        if best is not None and len(parts) >= len(best):
            return
        if start == len(token):
            if parts:
                best = tuple(parts)
            return
        for word in _VALID_WORDS:
            if token.startswith(word, start):
                parts.append(word)
                dfs(start + len(word), parts)
                parts.pop()

    dfs(0, [])
    return best


@lru_cache(maxsize=1024)
def _best_fuzzy_match(token: str) -> tuple[str, int] | None:
    best_word = None
    best_dist = None
    for word in _VALID_WORDS:
        dist = _edit_distance(token, word)
        max_allowed = 1 if len(word) <= 6 else 2
        if dist > max_allowed:
            continue
        if best_dist is None or dist < best_dist or (dist == best_dist and len(word) > len(best_word)):
            best_word = word
            best_dist = dist
    if best_word is None:
        return None
    return best_word, best_dist


def _repair_tokens(tokens: list[str]) -> tuple[list[str], bool]:
    repaired: list[str] = []
    used_fuzzy = False

    for token in tokens:
        if token in _VALID_WORDS:
            repaired.append(token)
            continue

        segmented = _segment_compound_token(token)
        if segmented is not None:
            repaired.extend(segmented)
            continue

        fuzzy = _best_fuzzy_match(token)
        if fuzzy is not None:
            repaired.append(fuzzy[0])
            used_fuzzy = True
            continue

        repaired.append(token)

    return repaired, used_fuzzy


def _parse_triplet(tokens: list[str]) -> Optional[int]:
    if not tokens:
        return 0

    idx = 0
    total = 0

    if idx < len(tokens) and tokens[idx] in _HUNDREDS:
        total += _HUNDREDS[tokens[idx]]
        idx += 1

    if idx < len(tokens) and tokens[idx] in _TEENS:
        total += _TEENS[tokens[idx]]
        idx += 1
        return total if idx == len(tokens) else None

    if idx < len(tokens) and tokens[idx] in _TENS:
        total += _TENS[tokens[idx]]
        idx += 1

    if idx < len(tokens) and tokens[idx] in _UNITS:
        total += _UNITS[tokens[idx]]
        idx += 1

    if idx != len(tokens):
        return None
    return total


def parse_number_text(text: str) -> ParseResult:
    normalized = normalize_number_text(text)
    if not normalized:
        return ParseResult(None, normalized, "", [])

    raw_tokens = normalized.split()
    repaired_tokens, used_fuzzy = _repair_tokens(raw_tokens)
    thousand_positions = [i for i, token in enumerate(repaired_tokens) if token in _THOUSAND]
    inferred_thousand = False

    value: Optional[int] = None

    if len(thousand_positions) == 1:
        pos = thousand_positions[0]
        left = repaired_tokens[:pos]
        right = repaired_tokens[pos + 1:]
        left_value = 1 if not left else _parse_triplet(left)
        right_value = _parse_triplet(right)
        if left_value is not None and 1 <= left_value <= 999 and right_value is not None:
            value = left_value * 1000 + right_value
    elif len(thousand_positions) == 0 and len(repaired_tokens) >= 3:
        candidates: list[int] = []
        for split in range(1, len(repaired_tokens)):
            left_value = _parse_triplet(repaired_tokens[:split])
            right_value = _parse_triplet(repaired_tokens[split:])
            if left_value is None or right_value is None:
                continue
            if 1 <= left_value <= 999:
                candidates.append(left_value * 1000 + right_value)
        if len(set(candidates)) == 1:
            value = candidates[0]
            inferred_thousand = True

    if value is None or not (1000 <= value <= 999999):
        return ParseResult(None, normalized, " ".join(repaired_tokens), repaired_tokens, used_fuzzy, inferred_thousand)

    return ParseResult(
        value=value,
        normalized_text=normalized,
        canonical_text=" ".join(repaired_tokens),
        repaired_tokens=repaired_tokens,
        used_fuzzy=used_fuzzy,
        inferred_thousand=inferred_thousand,
    )


def words_to_num(text: str) -> Optional[int]:
    """
    Parse Russian number words to integer.
    Returns None if parsing fails.
    """
    return parse_number_text(text).value


# --------------------------------------------------------------------------- #
#  Vocabulary for CTC                                                         #
# --------------------------------------------------------------------------- #

def build_vocab() -> tuple[dict[str, int], dict[int, str]]:
    """Собираем алфавит из всех слов, которые есть в наших словарях."""
    chars: set[str] = set()
    
    all_source_dicts = [_UNITS, _TEENS, _TENS, _HUNDREDS, _THOUSAND]
    for d in all_source_dicts:
        words = d.keys() if isinstance(d, dict) else d
        for word in words:
            for char in word:
                chars.add(char)
    
    chars.add(" ")
    sorted_chars = sorted(list(chars))

    char2idx = {c: i + 1 for i, c in enumerate(sorted_chars)}
    idx2char = {i + 1: c for i, c in enumerate(sorted_chars)}
    idx2char[0] = "<blank>"
    
    return char2idx, idx2char

_VOCAB: tuple[dict, dict] | None = None


def get_vocab() -> tuple[dict[str, int], dict[int, str]]:
    global _VOCAB
    if _VOCAB is None:
        _VOCAB = build_vocab()
    return _VOCAB


def encode_text(text: str, char2idx: dict[str, int]) -> list[int]:
    """Convert text string to list of indices (unknown chars skipped)."""
    return [char2idx[c] for c in text if c in char2idx]


def decode_indices(indices: list[int], idx2char: dict[int, str]) -> str:
    """Convert list of indices to string, ignoring blank (0)."""
    return "".join(idx2char.get(i, "") for i in indices if i != 0)


if __name__ == "__main__":
    # smoke test
    for n in [1000, 53583, 172306, 591872, 999999]:
        words = num_to_words(n)
        back = words_to_num(words)
        status = "Well done!" if back == n else f"Pupupu, got {back}"
        print(f"{n:>7} → '{words}' → {back} {status}")

    c2i, i2c = get_vocab()
    print(f"\nVocab size (incl. blank): {len(c2i) + 1}")
    print(f"Chars: {''.join(sorted(c2i.keys()))!r}")
