"""Per-language text sampling for synthetic Go book pages.

We only need text that *looks* right around the board diagrams — the
downstream detectors don't read it. For CJK languages we sample random real
characters from appropriate Unicode ranges. For Latin languages we draw from
a small bundled word list. All sampling is seeded for reproducibility.

Each language also advertises a system font that covers its script.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Language:
    code: str
    font_path: str
    # Token generator: returns a list of "words" (atomic text units) of
    # approximately the requested length.
    sample: callable  # type: ignore[type-arg]


# ---- CJK character pools ---------------------------------------------------

# Hangul syllables block.
_HANGUL = list(range(0xAC00, 0xD7A4))
# Hiragana (skip control chars).
_HIRAGANA = list(range(0x3041, 0x3097))
# Katakana.
_KATAKANA = list(range(0x30A1, 0x30FB))
# Most common Hanzi: sample from the CJK Unified Ideographs block's common
# range. Real frequency tables would be better, but uniform sampling over
# 4E00-9FA0 gets us a believable visual mix of strokes and densities.
_HANZI = list(range(0x4E00, 0x9FA0))


def _hangul_words(n: int, rng: random.Random) -> list[str]:
    # Korean books typically show 1-4 syllable words separated by spaces.
    out: list[str] = []
    for _ in range(n):
        length = rng.choices([1, 2, 3, 4], weights=[1, 4, 3, 1])[0]
        word = "".join(chr(rng.choice(_HANGUL)) for _ in range(length))
        out.append(word)
    return out


def _japanese_words(n: int, rng: random.Random) -> list[str]:
    # Japanese text mixes hiragana + katakana + kanji; no spaces between
    # "words" in running text, but books still have punctuation and line
    # breaks. Emit runs of 2-8 mixed characters as our visual "word".
    out: list[str] = []
    for _ in range(n):
        length = rng.randint(2, 8)
        chars = []
        for _ in range(length):
            pool = rng.choices(
                [_HIRAGANA, _KATAKANA, _HANZI], weights=[5, 1, 3],
            )[0]
            chars.append(chr(rng.choice(pool)))
        out.append("".join(chars))
    return out


def _chinese_words(n: int, rng: random.Random) -> list[str]:
    # Chinese text has no inter-word spaces either. Emit runs of 2-5 hanzi.
    out: list[str] = []
    for _ in range(n):
        length = rng.randint(2, 5)
        out.append("".join(chr(rng.choice(_HANZI)) for _ in range(length)))
    return out


# ---- Latin word lists ------------------------------------------------------

_EN_WORDS = (
    "the of and to a in that is was he for it with as his on be at by this "
    "had not are but from or have an they which one you were her all she there "
    "problem black white stone corner edge center diagram move answer next turn"
).split()

_FR_WORDS = (
    "le la les un une de du des et ou ne pas plus aussi que ce cet cette "
    "dans sur pour avec par sans sous contre bien très peu tout tous toute toutes "
    "problème noir blanc pierre coin bord centre diagramme coup réponse tour"
).split()

_DE_WORDS = (
    "der die das ein eine einen und oder nicht aber auch noch wenn dann als "
    "in auf mit für von zu bei über unter neben zwischen sehr viel wenig "
    "problem schwarz weiß stein ecke kante mitte diagramm zug antwort nächste"
).split()

_ES_WORDS = (
    "el la los las un una unos unas y o no ni pero también más menos muy poco "
    "en de por para con sin sobre bajo entre hacia desde hasta todo toda todos "
    "problema negro blanco piedra esquina borde centro diagrama movimiento respuesta"
).split()


def _latin_sample(words: tuple[str, ...]):
    def _sample(n: int, rng: random.Random) -> list[str]:
        return [rng.choice(words) for _ in range(n)]
    return _sample


# ---- font paths (macOS) ----------------------------------------------------

_FONT_KO = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
_FONT_JA = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"
_FONT_ZH = "/System/Library/Fonts/Supplemental/Songti.ttc"
_FONT_LATIN = "/System/Library/Fonts/Supplemental/Times New Roman.ttf"
_FONT_LATIN_FALLBACK = "/System/Library/Fonts/Helvetica.ttc"


LANGUAGES: dict[str, Language] = {
    "ko": Language("ko", _FONT_KO, _hangul_words),
    "ja": Language("ja", _FONT_JA, _japanese_words),
    "zh": Language("zh", _FONT_ZH, _chinese_words),
    "en": Language("en", _FONT_LATIN, _latin_sample(tuple(_EN_WORDS))),
    "fr": Language("fr", _FONT_LATIN, _latin_sample(tuple(_FR_WORDS))),
    "de": Language("de", _FONT_LATIN, _latin_sample(tuple(_DE_WORDS))),
    "es": Language("es", _FONT_LATIN, _latin_sample(tuple(_ES_WORDS))),
}

LATIN_FONT_FALLBACK = _FONT_LATIN_FALLBACK


def make_paragraph(lang: Language, word_count: int, rng: random.Random) -> str:
    """Return a paragraph of approximately `word_count` words.

    CJK scripts use no inter-word spaces in running text; Latin scripts join
    words with spaces. Sentences are capped with '.' at random intervals.
    """
    words = lang.sample(word_count, rng)
    if lang.code in ("ko", "en", "fr", "de", "es"):
        text = " ".join(words)
    else:  # ja, zh — no spaces
        text = "".join(words)
    return text
