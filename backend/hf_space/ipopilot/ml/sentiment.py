"""Model 3 — Financial news sentiment with FinBERT (ProsusAI/finbert).

Each headline gets P(positive), P(negative), P(neutral) from FinBERT plus a coarse topic
tag (earnings / regulatory / governance / industry / ipo-demand / other). Per-IPO
aggregates computed from headlines published *before* the decision time are fed to the
listing-return model as features.
"""
from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Iterable, Sequence

from ..config import settings

log = logging.getLogger(__name__)

TOPICS = {
    "regulatory": r"\b(sebi|probe|penalt|notice|ban|regulat|show[- ]cause|ed raid|tax demand|litigation|court)\b",
    "governance": r"\b(promoter|fraud|resign|auditor|governance|related[- ]party|pledge|whistle)\b",
    "earnings": r"\b(profit|revenue|earnings|ebitda|margin|loss|results|q[1-4]|fy\d{2}|growth)\b",
    "ipo_demand": r"\b(subscri|gmp|grey market|anchor|bids?|oversubscri|allotment|listing)\b",
    "industry": r"\b(sector|industry|demand|outlook|market share|competition|tariff|policy)\b",
}


def topic_of(title: str) -> str:
    t = title.lower()
    for name, pat in TOPICS.items():
        if re.search(pat, t):
            return name
    return "other"


@lru_cache(maxsize=1)
def _pipeline():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    name = settings.sentiment_model
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    labels = [model.config.id2label[i].lower() for i in range(model.config.num_labels)]
    log.info("Loaded sentiment model %s on %s labels=%s", name, device, labels)
    return tok, model, device, labels


def score_texts(texts: Sequence[str], batch_size: int = 32) -> list[dict[str, float]]:
    import torch

    if not texts:
        return []
    tok, model, device, labels = _pipeline()
    out: list[dict[str, float]] = []
    for i in range(0, len(texts), batch_size):
        batch = list(texts[i:i + batch_size])
        enc = tok(batch, padding=True, truncation=True, max_length=96, return_tensors="pt").to(device)
        with torch.no_grad():
            probs = torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()
        for p in probs:
            d = {lab: float(v) for lab, v in zip(labels, p)}
            out.append({"pos": d.get("positive", 0.0), "neg": d.get("negative", 0.0), "neu": d.get("neutral", 0.0)})
    return out


def score_items(items: Iterable) -> None:
    """Score NewsItem ORM objects in place (sets p_pos / p_neg / p_neu / topic)."""
    items = list(items)
    try:
        scores = score_texts([it.title for it in items])
    except Exception as e:  # model download failure etc. — keep the headlines, skip scores
        log.warning("Sentiment scoring unavailable: %s", e)
        scores = [None] * len(items)
    for it, sc in zip(items, scores):
        it.topic = topic_of(it.title)
        if sc:
            it.p_pos, it.p_neg, it.p_neu = sc["pos"], sc["neg"], sc["neu"]
