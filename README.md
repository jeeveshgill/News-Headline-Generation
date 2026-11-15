# News Headline Generation using NLP (BCA 5th Semester)

Generates a short headline from the body of a news article and compares **extractive** methods
(Lead-1, TF-IDF, TF-IDF + proper-noun boost, TextRank) with an **abstractive** transformer (fine-tuned T5-small).

## Folder structure
```
dataset/        sample_news.csv (10-row DEMO only) + your real dataset (news.csv)
notebooks/      analysis.ipynb  - step-by-step walkthrough
src/            preprocessing.py, model.py, transformer_model.py, evaluation.py, prediction.py, run_experiments.py
app/            app.py (Streamlit demo)
results/        model_comparison.csv, dataset_stats.csv, graphs/
outputs/        predictions.csv, error_analysis.csv
```

## Dataset (you must download it - it is not bundled)
Recommended: **Inshorts News dataset on Kaggle** (headline + ~60-word article). Any CSV with an article-text
column and a headline column works (e.g. "News Summary" on Kaggle, CNN/DailyMail `highlights`).
Put it at `dataset/news.csv`, open it once, and note the exact column names. Cite the source, number of records,
fields and licence in your report.

## Run
```bash
pip install -r requirements.txt
python src/run_experiments.py --data dataset/news.csv --text-col Short --headline-col Headline
# optional abstractive model (use Google Colab GPU; ~20k training articles, 3 epochs)
python src/transformer_model.py --data dataset/news.csv --text-col Short --headline-col Headline
python src/run_experiments.py --data dataset/news.csv --text-col Short --headline-col Headline --t5
# demo app
NEWS_CSV=dataset/news.csv TEXT_COL=Short HEAD_COL=Headline streamlit run app/app.py
```
Everything you report (tables, graphs, examples) must come from these runs - guidelines forbid fabricated results.

## Pipeline
Cleaning (HTML/URL/whitespace, duplicates, short rows) -> sentence splitting -> tokenisation -> lowercasing ->
stop-word removal -> lemmatisation -> TF-IDF features -> sentence ranking (TF-IDF / TextRank) -> rule-based
compression to <= 12 words -> evaluation. Transformer path: T5 tokenizer -> fine-tune -> beam-search decoding.
Split 80/10/10 (seed 42). IDF is learnt on training data only.

## Evaluation
ROUGE-1/2/L F1 (own implementation in `src/evaluation.py`, no stemming - numbers can differ slightly from the
`rouge_score` package; say which you used), average headline length, plus manual error analysis of the 20 worst cases
(`outputs/error_analysis.csv`). Optional: BERTScore, small human rating (fluency / informativeness, 1-5).

## Notes / honesty
* `sample_news.csv` is a tiny hand-written demo for smoke-testing; never report results from it.
* The code was smoke-tested on the demo file only. `transformer_model.py` and `app/app.py` need torch/transformers/streamlit
  and were not executed in the authoring environment - run them once and fix any version issues.
* Ethics: headline generators can hallucinate facts (abstractive) or create clickbait; keep a human editor in the loop.
