"""Abstractive headline generation: fine-tune t5-small (use a Google Colab GPU for speed).

    python src/transformer_model.py --data dataset/news.csv --text-col Short --headline-col Headline
"""
import argparse
import os

import torch
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, T5TokenizerFast

PREFIX = "generate headline: "
MODEL_DIR = "results/t5_headline"


def generate(texts, model_dir=MODEL_DIR, batch=32, max_len=24, beams=4):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = T5TokenizerFast.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir).to(dev).eval()
    out = []
    for i in range(0, len(texts), batch):
        enc = tok([PREFIX + t for t in texts[i:i + batch]], max_length=256, truncation=True,
                  padding=True, return_tensors="pt").to(dev)
        with torch.no_grad():
            ids = model.generate(**enc, max_length=max_len, num_beams=beams, early_stopping=True)
        out += tok.batch_decode(ids, skip_special_tokens=True)
    return out


def train(train_df, val_df, epochs=3, lr=3e-4, batch=16, max_train=20000, base="t5-small"):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = T5TokenizerFast.from_pretrained(base)
    model = T5ForConditionalGeneration.from_pretrained(base).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    tr = train_df.sample(min(max_train, len(train_df)), random_state=42)

    def collate(rows):
        x = tok([PREFIX + r[0] for r in rows], max_length=256, truncation=True, padding=True, return_tensors="pt")
        y = tok([r[1] for r in rows], max_length=24, truncation=True, padding=True, return_tensors="pt").input_ids
        y[y == tok.pad_token_id] = -100  # ignore padding in the loss
        return x, y

    rows = list(zip(tr.text, tr.headline))
    vrows = list(zip(val_df.text[:500], val_df.headline[:500]))
    for ep in range(epochs):
        model.train()
        for x, y in DataLoader(rows, batch_size=batch, shuffle=True, collate_fn=collate):
            loss = model(**x.to(dev), labels=y.to(dev)).loss
            loss.backward(); opt.step(); opt.zero_grad()
        model.eval(); vl, nb = 0.0, 0
        with torch.no_grad():
            for x, y in DataLoader(vrows, batch_size=batch, collate_fn=collate):
                vl += model(**x.to(dev), labels=y.to(dev)).loss.item(); nb += 1
        print(f"epoch {ep + 1}: last train loss {loss.item():.3f} | val loss {vl / max(nb, 1):.3f}")
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_pretrained(MODEL_DIR); tok.save_pretrained(MODEL_DIR)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from preprocessing import load_dataset, split_dataset
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--text-col", default="text")
    ap.add_argument("--headline-col", default="headline"); ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-train", type=int, default=20000)
    a = ap.parse_args()
    tr, va, _ = split_dataset(load_dataset(a.data, a.text_col, a.headline_col))
    train(tr, va, epochs=a.epochs, max_train=a.max_train)
