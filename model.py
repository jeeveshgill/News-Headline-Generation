"""Extractive headline generators: Lead-1, TF-IDF sentence scoring (+ proper-noun boost), TextRank."""
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from preprocessing import STOP_WORDS, analyzer, split_sentences

MAX_WORDS = 12


def compress_sentence(sentence: str, max_words: int = MAX_WORDS) -> str:
    """Turn a full sentence into headline-length text (rule-based compression)."""
    s = re.sub(r"\([^)]*\)", "", sentence)                       # drop parentheticals
    s = re.sub(r"^(According to [^,]+|As per [^,]+),\s*", "", s, flags=re.I)
    parts = [p.strip() for p in re.split(r"[,;:]\s+|\s[-–—]\s", s) if p.strip()]
    clause = next((p for p in parts if len(p.split()) >= 4), s)  # first clause with >= 4 words
    words = clause.split()[:max_words]
    while words and words[-1].lower().strip(".,;:") in STOP_WORDS:  # no dangling "the", "of"...
        words.pop()
    h = " ".join(words).strip(" .,;:")
    return h[:1].upper() + h[1:] if h else sentence[:60]


def fit_vectorizer(train_texts) -> TfidfVectorizer:
    """IDF is learnt from every training sentence."""
    sents = [s for t in train_texts for s in split_sentences(t)]
    vec = TfidfVectorizer(analyzer=analyzer, sublinear_tf=True, norm=None)
    return vec.fit(sents)


class LeadHeadline:
    """Baseline: compress the first sentence (news puts key facts first)."""
    name = "Lead-1 (baseline)"

    def generate(self, article: str) -> str:
        sents = split_sentences(article) or [article]
        return compress_sentence(sents[0])


class TfidfHeadline:
    """Score each sentence by mean TF-IDF weight of its words; compress the best one."""

    def __init__(self, vectorizer, entity_boost: float = 0.0):
        self.vec, self.boost = vectorizer, entity_boost
        self.name = "TF-IDF" + (" + proper-noun boost" if entity_boost else "")

    def generate(self, article: str) -> str:
        sents = split_sentences(article) or [article]
        X = self.vec.transform(sents)
        n_terms = np.maximum((X > 0).sum(axis=1).A1, 1)
        scores = X.sum(axis=1).A1 / n_terms
        if self.boost:  # headlines are full of names: reward capitalised words inside the sentence
            for i, s in enumerate(sents):
                w = s.split()[1:]
                scores[i] *= 1 + self.boost * (sum(x[:1].isupper() for x in w) / max(len(w), 1))
        return compress_sentence(sents[int(np.argmax(scores))])


class TextRankHeadline:
    """Graph ranking: sentences are nodes, TF-IDF cosine similarity gives the edges."""
    name = "TextRank"

    def __init__(self, vectorizer, damping=0.85, iters=50):
        self.vec, self.d, self.iters = vectorizer, damping, iters

    def generate(self, article: str) -> str:
        sents = split_sentences(article) or [article]
        n = len(sents)
        if n == 1:
            return compress_sentence(sents[0])
        X = normalize(self.vec.transform(sents))
        sim = (X @ X.T).toarray()
        np.fill_diagonal(sim, 0)
        row = sim.sum(axis=1, keepdims=True)
        P = np.divide(sim, row, out=np.full_like(sim, 1.0 / n), where=row > 0)
        r = np.full(n, 1.0 / n)
        for _ in range(self.iters):
            r = (1 - self.d) / n + self.d * P.T @ r
        return compress_sentence(sents[int(np.argmax(r))])
