"""
Text normalization and denormalization for Russian spoken numbers.
Range: 1_000 .. 999_999
"""
from __future__ import annotations
import re
from typing import Optional

# --------------------------------------------------------------------------- #
#  Forward: integer → Russian words (used to build CTC targets)               #
# --------------------------------------------------------------------------- #

try:
    from num2words import num2words as _n2w

    def num_to_words(n: int) -> str:
        """Convert integer to lower-case Russian words string."""
        return _n2w(n, lang="ru").lower().replace("-", " ").replace("  ", " ").strip()

except ImportError:
    raise ImportError("Install num2words: pip install num2words")


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


def words_to_num(text: str) -> Optional[int]:
    """
    Parse Russian number words to integer.
    Returns None if parsing fails.
    """
    text = text.lower().strip()
    # strip punctuation
    text = re.sub(r"[^а-яё ]", " ", text)
    words = text.split()

    result = 0
    chunk = 0   # accumulator for the current "order" (thousands part)

    i = 0
    while i < len(words):
        w = words[i]

        if w in _HUNDREDS:
            chunk += _HUNDREDS[w]
        elif w in _TEENS:
            chunk += _TEENS[w]
        elif w in _TENS:
            chunk += _TENS[w]
        elif w in _UNITS:
            chunk += _UNITS[w]
        elif w in _THOUSAND:
            if chunk == 0:
                chunk = 1
            result += chunk * 1_000
            chunk = 0
        # ignore unknown tokens (noise from ASR errors)

        i += 1

    result += chunk
    return result if result > 0 else None


# --------------------------------------------------------------------------- #
#  Vocabulary for CTC                                                         #
# --------------------------------------------------------------------------- #

def build_vocab() -> tuple[dict[str, int], dict[int, str]]:
    """Собираем алфавит из всех слов, которые есть в наших словарях."""
    chars: set[str] = set()
    
    # Собираем все буквы из всех возможных слов
    all_source_dicts = [_UNITS, _TEENS, _TENS, _HUNDREDS, _THOUSAND]
    for d in all_source_dicts:
        # d может быть словарем или множеством
        words = d.keys() if isinstance(d, dict) else d
        for word in words:
            for char in word:
                chars.add(char)
    
    chars.add(" ") # Не забываем пробел
    sorted_chars = sorted(list(chars))

    # 0 = blank (CTC), 1.. = буквы
    char2idx = {c: i + 1 for i, c in enumerate(sorted_chars)}
    idx2char = {i + 1: c for i, c in enumerate(sorted_chars)}
    idx2char[0] = "<blank>"
    
    return char2idx, idx2char

# Singleton vocab (lazy)
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
        status = "✓" if back == n else f"✗ got {back}"
        print(f"{n:>7} → '{words}' → {back} {status}")

    c2i, i2c = get_vocab()
    print(f"\nVocab size (incl. blank): {len(c2i) + 1}")
    print(f"Chars: {''.join(sorted(c2i.keys()))!r}")
