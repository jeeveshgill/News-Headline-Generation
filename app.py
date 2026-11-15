"""Streamlit demo.  Run:  streamlit run app/app.py"""
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from preprocessing import load_dataset, split_dataset
from prediction import build_generators, predict_all

DATA = os.environ.get("NEWS_CSV", "dataset/news_summary.csv")
TEXT_COL, HEAD_COL = os.environ.get("TEXT_COL", "text"), os.environ.get("HEAD_COL", "headlines")


@st.cache_resource
def load():
    train, _, _ = split_dataset(load_dataset(DATA, TEXT_COL, HEAD_COL))
    return build_generators(train.text)


st.title("News Headline Generator")
st.caption("Extractive (Lead-1, TF-IDF, TextRank) and optional abstractive (T5) headline generation")
article = st.text_area("Paste a news article", height=220)
use_t5 = st.checkbox("Include T5 model (needs results/t5_headline)")
if st.button("Generate headline") and article.strip():
    for name, headline in predict_all(article, load(), use_t5).items():
        st.markdown(f"**{name}:** {headline}")
