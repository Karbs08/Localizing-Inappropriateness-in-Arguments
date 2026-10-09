# Baseline experiments

This directory contains the two model-independent reference strategies. Both select spans without using classifier attributions, but use the fixed classifier to assess the masking effect of those spans. See the [project README](../README.md) for the complete protocol.

## Notebooks

| Notebook | Selection strategy | Final thesis configuration | Output directory |
|---|---|---|---|
| [random_baseline.ipynb](random_baseline.ipynb) | Compares contiguous random spans with distributed random word-budget selection. | `multi_word_budgeted`, 25 words | `results/random_baseline_results/` |
| [tf-idf_baseline.ipynb](tf-idf_baseline.ipynb) | Selects high-scoring TF-IDF terms, finds their occurrences, expands word context, and merges spans. | `top_k=8`, `window_size=2` | `results/tfidf_baseline_results/` |

Paths are relative to the repository root. Both notebooks load the canonical prepared dataset through `src.data`, select configurations on the **873 train/validation true positives**, and apply their selected settings to held-out test arguments. Masking is fixed before configuration search.

## Run order

1. Run `make init` and start Jupyter with the project environment.
2. Reproduce the preliminary operator comparison in [evaluation](../evaluation/README.md).
3. Run either baseline notebook independently; neither requires the other baseline's outputs.
4. Retain the selected configuration and final exports for cross-method evaluation and survey preparation.

The TF-IDF vectorizer is fitted on all **1,753 train/validation texts**, with test texts only transformed. Its ranking evaluation uses word occurrences, without the context expansion applied to final spans.

Random uses **five draws per argument/configuration** during development, aggregated within each argument. Final evaluation uses **one reproducible draw per argument**, without choosing the most effective draw.

## Important exports

| Export, relative to the method's output directory | Purpose |
|---|---|
| `random_train_val_tp_config_summary.csv` | Random development configuration summary. |
| `random_selected_config_manual.json` | Saved selected Random settings. |
| `random_best_config_test_per_argument.csv` | Fixed Random test explanations and perturbation results used downstream. |
| `random_best_config_test_tp_per_argument.csv` | True-positive test subset. |
| `random_best_config_confusion_summary.csv` | Random summary by split/confusion type. |
| `development_config_selection/tfidf_development_config_summary.csv` | TF-IDF development configuration summary. |
| `final_test_all_window2_topk8/tfidf_final_test_all_window2_topk8_argument_level.csv` | Final TF-IDF test explanations used downstream. |
| `final_test_all_window2_topk8/tfidf_final_test_all_window2_topk8_confusion_summary.csv` | Final TF-IDF summary by confusion type. |

Additional exports include span-level tables, JSONL records, plots, and Random sample-level tables. TF-IDF also writes ranking results under `aopc_ranking_evaluation/` and uses a `cache/` directory.

## Before rerunning

Random currently contains an active read of `random_train_val_tp_config_summary.csv` through a local `/Users/karsten/...` path. Replace that read with `OUTPUT_DIR / "random_train_val_tp_config_summary.csv"` when running elsewhere.

The final settings are selected from development results. If they differ in a new run, update downstream filenames consistently. Keep all offsets tied to the shared normalized text and do not select configurations using test results.

Continue with [evaluation](../evaluation/README.md), or inspect the full [results guide](../results/README.md).
