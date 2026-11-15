"""Build all generators once and produce headlines for new text (used by the app)."""
import os

from model import LeadHeadline, TextRankHeadline, TfidfHeadline, fit_vectorizer
from preprocessing import clean_text


def build_generators(train_texts):
    vec = fit_vectorizer(train_texts)
    return [LeadHeadline(), TfidfHeadline(vec), TfidfHeadline(vec, entity_boost=1.0), TextRankHeadline(vec)]


def predict_all(article, generators, use_t5=False):
    article = clean_text(article)
    out = {g.name: g.generate(article) for g in generators}
    if use_t5 and os.path.isdir("results/t5_headline"):
        from transformer_model import generate
        out["T5 (abstractive)"] = generate([article])[0]
    return out
