import os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from evaluation import rouge_n

df = pd.read_csv("outputs/predictions.csv")
systems = ["Lead-1 (baseline)", "TextRank", "T5 (abstractive)"]
for s in systems:
    df["R1_" + s] = [rouge_n(r, h, 1)[2] for r, h in zip(df.headline, df[s])]

def show(title, rows):
    print("\n=====", title, "=====")
    for _, r in rows.iterrows():
        print("REF     :", r.headline)
        for s in systems:
            print(f"{s[:8]:8s}:", r[s], f"(R1={r['R1_' + s]:.2f})")
        print()

t5, lead = "R1_T5 (abstractive)", "R1_Lead-1 (baseline)"
show("T5 GOOD (highest T5 score)", df.nlargest(3, t5))
show("T5 FAILS (lowest T5 score)", df.nsmallest(3, t5))
show("LEAD-1 BEATS T5 (biggest gap)", df.assign(gap=df[lead] - df[t5]).nlargest(3, "gap"))