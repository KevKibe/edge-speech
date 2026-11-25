"""Language-agnostic text normalization applied to BOTH reference and prediction.

Modeled on Whisper's BasicTextNormalizer, with one change: combining marks
(Unicode category M) are kept. Whisper turns them into spaces, which shreds
Indic, Thai, and other abugida scripts into fragments.
"""

from __future__ import annotations

import re
import unicodedata

_BRACKETS = re.compile(r"[<\[][^>\]]*[>\]]")  # <noise>, [laughter]
_PARENS = re.compile(r"\([^)]*\)")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = _BRACKETS.sub(" ", text)
    text = _PARENS.sub(" ", text)
    # Punctuation (P*) and symbols (S*) become spaces; letters, digits, marks stay.
    text = "".join(" " if unicodedata.category(ch)[0] in "PS" else ch for ch in text)
    return " ".join(text.split())
