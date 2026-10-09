# Experimental results

This directory holds generated experiment outputs, caches, checkpoints, comparison tables, and figures. Results are produced by the [baselines](../baselines/README.md), [methods](../methods/README.md), and [evaluation notebooks](../evaluation/README.md). Human-study analysis is stored separately in `human_study/survey_results/`.

All paths below are relative to the repository root. Large outputs are excluded by the result folders' Git ignore rules; a fresh checkout does not contain the final CSVs required by the evaluation notebooks.

## Directory map

| Directory | Producer and contents |
|---|---|
| `results/random_baseline_results/` | Random development draws, selected settings, final explanations, summaries, and plots. |
| `results/tfidf_baseline_results/` | TF-IDF development search, final test spans, cache, ranking evaluation, and plots. |
| `results/attention_results/` | Attention cache, development selection, final spans, ranking evaluation, and plots. |
| `results/ig_results/` | IG attribution cache, development selection, final spans, ranking evaluation, and plots. |
| `results/shap_results/` | SHAP cache, configuration search, final spans, ranking evaluation, and plots. |
| `results/mil_results/` | Coarse/refined run folders, per-configuration checkpoints, validation predictions, and selected-model exports. |
| `results/llm_reference/` | Fixed annotation sample, generated/validated spans, and reference perturbation results. |
| `results/ablation_comparison/` | Preliminary paired masking/deletion comparison: tables, plots, cache, and lightweight MIL checkpoints. |
| `results/evaluation/comparison/` | Shared six-method perturbation comparison and statistical/diagnostic outputs. |
| `results/evaluation/methods_vs_llm/` | Matched-reference token overlap, perturbation comparisons, and tests. |
| `results/evaluation/mil/` | MIL document-classification tables and figures from `mil_document_level.ipynb`. |
| `results/evaluation/tables/` and `plots/` | Additional general outputs written by `methods_vs_llm.ipynb`. |

## Final argument-level inputs used downstream

These are the paths currently expected by the comparison notebooks for the thesis configurations:

| Source | CSV path |
|---|---|
| Random | `results/random_baseline_results/random_best_config_test_per_argument.csv` |
| TF-IDF | `results/tfidf_baseline_results/final_test_all_window2_topk8/tfidf_final_test_all_window2_topk8_argument_level.csv` |
| Attention | `results/attention_results/attention_results_final/attention_final_all_splits_rollout_q0.4_window0_argument_level.csv` |
| IG | `results/ig_results/ig_results_final/ig_final_all_splits_q0.4_window3_argument_level.csv` |
| SHAP | `results/shap_results/shap_results_final/shap_final_test_q0.5_window0_argument_level.csv` |
| MIL | `results/mil_results/mil_span_run_singles/mil_results_final/mil_final_all_splits_pool=topk_noisy_or__topk=3__spans=15__stride=4__freeze=True_argument_level.csv` |
| LLM | `results/llm_reference/llm_test_tp_spans_argument_level.csv` |

For TF-IDF, Attention, IG, SHAP, and MIL, comparison notebooks also read the corresponding `_confusion_summary.csv` in the same final directory. Random uses `random_best_config_confusion_summary.csv`; the LLM comparison additionally reads a reference summary.

## Reading the files

- **Argument-level exports** contain selected spans, original/perturbed outputs, coverage, and other per-argument metadata. Alignment uses `global_row_id`.
- **Span-level exports** contain individual regions or candidate scores. MIL candidate-score files are distinct from its selected explanation spans.
- **Development summaries** compare configurations and should not be reported as final held-out performance.
- **Final files** represent fixed configurations; `_all_splits_` files require filtering by `split` and `confusion_type` for a test-TP comparison. SHAP's final export covers test data.
- **AOPC exports** use `<prefix>_raw.csv`, `_step_summary.csv`, `_curve_summary.csv`, `_per_argument.csv`, and `_global_summary.csv` within the relevant method's `aopc_ranking_evaluation/` directory.
- **Caches and checkpoints** support reuse within a compatible run; keep them tied to the same normalized text, model/tokenizer, and settings.

The main test analysis has 438 arguments and 186 TPs; reference agreement uses 100 annotated test TPs. Inspect shared-ID coverage before comparing outputs. Human-facing word-expanded highlights are separate from the raw automatic intervals in these files.