"""Normalized EM for response-only predictions; see docs/PAPER_PARITY.md."""
from __future__ import annotations
import re
import string
from collections.abc import Sequence


def normalize_answer(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError('Answers must be strings.')
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    return ' '.join(text.split())


def extract_answer(output: str, *, tagged: bool) -> str:
    if not isinstance(output, str):
        raise TypeError('Model output must be a string.')
    if not tagged:
        return output.strip()
    matches = re.findall(r'<answer>(.*?)</answer>', output, flags=re.DOTALL)
    return matches[-1].strip() if matches else ''


def exact_match(prediction: str, answers: Sequence[str]) -> int:
    if isinstance(answers, (str, bytes)) or not answers:
        raise ValueError('answers must be a nonempty sequence of strings.')
    if any(not isinstance(a, str) or not a.strip() for a in answers):
        raise ValueError('Gold answers must be nonempty strings.')
    if not isinstance(prediction, str):
        raise TypeError('Prediction must be a string.')
    if not prediction.strip():
        return 0
    pred = normalize_answer(prediction)
    return int(any(pred == normalize_answer(a) for a in answers))
