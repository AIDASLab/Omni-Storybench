
import gc
from typing import Iterable, List, Tuple

import torch
from bert_score import BERTScorer


_BERT_SCORER_CACHE = {}


def _resolve_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_bert_scorer(language: str = "en", device: str | None = None) -> BERTScorer:
    resolved_device = device or _resolve_device()
    cache_key = (language, resolved_device)
    cached = _BERT_SCORER_CACHE.get(cache_key)
    if cached is not None:
        return cached

    scorer = BERTScorer(
        lang=language,
        device=resolved_device,
        rescale_with_baseline=False,
    )
    _BERT_SCORER_CACHE[cache_key] = scorer
    return scorer


def get_bert_scores(
    candidate_texts: Iterable[str],
    reference_texts: Iterable[str],
    language: str = "en",
) -> Tuple[List[float], List[float], List[float]]:
    cands = list(candidate_texts)
    refs = list(reference_texts)
    if len(cands) != len(refs):
        raise ValueError("candidate_texts and reference_texts must have the same length")
    if not cands:
        return [], [], []

    scorer = get_bert_scorer(language=language)
    precision, recall, f1 = scorer.score(cands, refs)
    return precision.tolist(), recall.tolist(), f1.tolist()


def get_bert_score(candidate_text, reference_text, language="en"):
    precision, recall, f1 = get_bert_scores(
        [candidate_text],
        [reference_text],
        language=language,
    )
    return precision[0], recall[0], f1[0]


def clear_bert_score_cache() -> None:
    for scorer in _BERT_SCORER_CACHE.values():
        model = getattr(scorer, "_model", None)
        if model is not None and hasattr(model, "cpu"):
            try:
                model.cpu()
            except Exception:
                pass

    _BERT_SCORER_CACHE.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
