"""Run the full pipeline: load -> clean -> split -> generate -> evaluate -> save tables, graphs, error analysis.

    python src/run_experiments.py --data dataset/news.csv --text-col Short --headline-col Headline [--t5]
"""
import argparse
import os
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from evaluation import evaluate, rouge_n
from preprocessing import load_dataset, split_dataset, tokenize, STOP_WORDS
from prediction import build_generators

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="dataset/sample_news.csv")
ap.add_argument("--text-col", default="text")
ap.add_argument("--headline-col", default="headline")
ap.add_argument("--max-test", type=int, default=5000)
ap.add_argument("--t5", action="store_true", help="also evaluate the fine-tuned T5 model")
a = ap.parse_args()
for d in ("results/graphs", "outputs"):
    os.makedirs(d, exist_ok=True)

# 1. data ---------------------------------------------------------------------------------------
df = load_dataset(a.data, a.text_col, a.headline_col)
train, val, test = split_dataset(df)
test = test.head(a.max_test)
stats = pd.DataFrame({"Records after cleaning": [len(df)], "Train": [len(train)], "Validation": [len(val)],
                      "Test used": [len(test)],
                      "Avg article words": [round(df.text.str.split().str.len().mean(), 1)],
                      "Avg headline words": [round(df.headline.str.split().str.len().mean(), 1)]})
stats.to_csv("results/dataset_stats.csv", index=False)
print(stats.T.to_string(header=False))

# 2. generate -----------------------------------------------------------------------------------
gens = build_generators(train.text)
preds = {g.name: [g.generate(t) for t in test.text] for g in gens}
if a.t5:
    from transformer_model import generate
    preds["T5 (abstractive)"] = generate(list(test.text))

# 3. evaluate -----------------------------------------------------------------------------------
table = pd.DataFrame({k: evaluate(test.headline, v) for k, v in preds.items()}).T
table.index.name = "Approach"
table.to_csv("results/model_comparison.csv")
print(table.to_string())

out = test.copy()
for k, v in preds.items():
    out[k] = v
out.to_csv("outputs/predictions.csv", index=False)

# 4. error analysis: 20 worst test cases of the best system ---------------------------------------
best = table["ROUGE-1 F1"].idxmax()
out["best_R1"] = [rouge_n(r, h)[2] for r, h in zip(out.headline, out[best])]
out.nsmallest(20, "best_R1")[["text", "headline", best, "best_R1"]].to_csv("outputs/error_analysis.csv", index=False)

# 5. graphs ---------------------------------------------------------------------------------------
ax = table[["ROUGE-1 F1", "ROUGE-2 F1", "ROUGE-L F1"]].plot.bar(figsize=(9, 5), rot=15)
ax.set_title("ROUGE F1 by approach"); ax.set_ylabel("Score"); plt.tight_layout()
plt.savefig("results/graphs/rouge_comparison.png", dpi=150); plt.close()

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
df.text.str.split().str.len().plot.hist(bins=30, ax=axs[0], title="Article length (words)")
df.headline.str.split().str.len().plot.hist(bins=15, ax=axs[1], title="Reference headline length (words)")
plt.tight_layout(); plt.savefig("results/graphs/length_distributions.png", dpi=150); plt.close()

words = Counter(w.lower() for h in df.headline for w in tokenize(h) if w.lower() not in STOP_WORDS)
top = pd.Series(dict(words.most_common(20)))[::-1]
top.plot.barh(figsize=(7, 6), title="Top 20 words in reference headlines"); plt.tight_layout()
plt.savefig("results/graphs/top_headline_words.png", dpi=150); plt.close()
print("Saved: results/, outputs/")
