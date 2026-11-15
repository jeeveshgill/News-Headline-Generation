"""Cleaning, sentence splitting, tokenisation, stop-word removal, lemmatisation, dataset loading."""
import html
import re

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split

try:  # NLTK is preferred; fall back gracefully if its data cannot be downloaded
    import nltk
    for _pkg in ("stopwords", "wordnet", "omw-1.4"):
        nltk.download(_pkg, quiet=True)
    from nltk.corpus import stopwords as _sw
    from nltk.stem import WordNetLemmatizer
    STOP_WORDS = set(_sw.words("english"))
    _lemma = WordNetLemmatizer().lemmatize
    _lemma("tests")  # raises LookupError if WordNet is missing
except Exception:
    STOP_WORDS = set(ENGLISH_STOP_WORDS)
    _lemma = lambda w: w

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")
_TOKEN = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*")


def clean_text(text) -> str:
    """Remove HTML, URLs and extra whitespace (case is kept: capitals help find names)."""
    text = html.unescape(str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> list:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def tokenize(text: str) -> list:
    return _TOKEN.findall(text)


def normalize_tokens(tokens, remove_stop=True, lemmatize=True) -> list:
    """Lowercase -> stop-word removal -> lemmatisation."""
    out = []
    for t in tokens:
        t = t.lower()
        if remove_stop and t in STOP_WORDS:
            continue
        out.append(_lemma(t) if lemmatize else t)
    return out


def analyzer(text: str) -> list:
    """Analyzer used by TfidfVectorizer."""
    return normalize_tokens(tokenize(text))


def load_dataset(path, text_col="text", headline_col="headline", min_text_words=15, min_headline_words=3):
    try:
        df = pd.read_csv(path)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")
    df = df[[text_col, headline_col]].rename(columns={text_col: "text", headline_col: "headline"})
    df = df.dropna()
    df["text"] = df["text"].map(clean_text)
    df["headline"] = df["headline"].map(clean_text)
    df = df.drop_duplicates(subset="text")
    df = df[(df.text.str.split().str.len() >= min_text_words) &
            (df.headline.str.split().str.len() >= min_headline_words)]
    return df.reset_index(drop=True)


def split_dataset(df, seed=42):
    """80 / 10 / 10 train / validation / test split."""
    train, rest = train_test_split(df, test_size=0.2, random_state=seed)
    val, test = train_test_split(rest, test_size=0.5, random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
