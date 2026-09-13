"""Small deterministic token estimation helpers."""

from __future__ import annotations

import math


TOKEN_ESTIMATOR_VERSION = "mixed-language-v1"


def estimate_tokens(text: str) -> int:
    """Estimate mixed-language token use without a provider tokenizer.

    ASCII-heavy source and JSON remain close to the traditional four
    characters per token heuristic. CJK text is counted conservatively at one
    character per token, while other Unicode text uses two characters per
    token.
    """

    if not text:
        return 0

    ascii_chars = 0
    cjk_chars = 0
    other_unicode_chars = 0
    for character in text:
        codepoint = ord(character)
        if codepoint < 128:
            ascii_chars += 1
        elif _is_cjk(codepoint):
            cjk_chars += 1
        else:
            other_unicode_chars += 1

    return (
        math.ceil(ascii_chars / 4)
        + cjk_chars
        + math.ceil(other_unicode_chars / 2)
    )


def _is_cjk(codepoint: int) -> bool:
    """Return whether one code point belongs to a CJK writing-system block."""

    return any(
        start <= codepoint <= end
        for start, end in (
            (0x2E80, 0x2FFF),
            (0x3040, 0x30FF),
            (0x3100, 0x312F),
            (0x3130, 0x318F),
            (0x31A0, 0x31BF),
            (0x31C0, 0x31EF),
            (0x3400, 0x4DBF),
            (0x4E00, 0x9FFF),
            (0xA960, 0xA97F),
            (0xAC00, 0xD7AF),
            (0xD7B0, 0xD7FF),
            (0xF900, 0xFAFF),
            (0x20000, 0x2FA1F),
        )
    )
