# Localization experiments

This directory contains the model-based span-localization experiments and the LLM silver-reference generator. See the [project README](../README.md) for the research context and complete experimental protocol.

## Notebooks and outputs

All output paths below are relative to the repository root.

| Notebook | What it does | Output directory |
|---|---|---|
| [attention_spans.ipynb](attention_spans.ipynb) | Compares final-layer CLS attention and attention rollout; selects token spans and evaluates masking effects and relevance rankings. | `results/attention_results/` |
| [integrated_gradients_spans.ipynb](integrated_gradients_spans.ipynb) | Computes embedding-layer Integrated Gradients for `LABEL_1`; selects positive model-token attributions and merges them into spans. | `results/ig_results/` |
| [shap_spans.ipynb](shap_spans.ipynb) | Caches SHAP segment attributions, selects positive evidence, and evaluates final spans and rankings. | `results/shap_results/` |
| [mil_spans.ipynb](mil_spans.ipynb) | Performs the coarse MIL search over grouped candidate-span lengths, training from document labels. | `results/mil_results/mil_span_run_span_groups/` |
| [mil_spans_singles.ipynb](mil_spans_singles.ipynb) | Performs the refined MIL search over individual span lengths and exports the selected model and explanations. | `results/mil_results/<RUN_NAME>/` |
| [llm_spans.ipynb](llm_spans.ipynb) | Generates and validates Gemini annotations for 100 fixed test true positives, then evaluates their masking effects. | `results/llm_reference/` |

Attention and IG export final files under `attention_results_final/` and `ig_results_final/`; SHAP uses `shap_results_final/`. Development summaries and cached scores remain separate from these final outputs. Ranking tables use `aopc_ranking_evaluation/`.

MIL stores each trained configuration under `validation_grid/<config_id>/`, including training history, validation predictions, span scores, and `model_checkpoint.pt`. Selected outputs, `best_config.json`, and `best_mil_model_checkpoint.pt` are written to `mil_results_final/` within the run directory.

## Running the experiments

1. Run `make init` from the repository root and select the project Jupyter kernel.
2. Reproduce the preliminary operator comparison in [evaluation](../evaluation/README.md); subsequent experiments use masking.
3. Run the required localization notebook. The coarse MIL search explores the search space; the refined notebook selects the thesis model and does not load coarse-search results as an input.
4. Inspect the selected configuration and exported files before running cross-method evaluation or preparing the survey.

All notebooks load `data/processed/appropriateness_prepared.parquet` through `src.data`. Attention, IG, and SHAP select configurations on the 873 train/validation true positives. MIL trains on the training split and selects on validation. Test data is reserved for final evaluation.

| Method | Final thesis settings |
|---|---|
| Attention | `rollout`, `quantile=0.40`, `window_size=0` |
| Integrated Gradients | `quantile=0.40`, `window_size=3` |
| SHAP | `quantile=0.50`, `window_size=0` |
| MIL | `topk_noisy_or`, `topk=3`, `spans=15`, `stride=4`, frozen encoder |

The notebooks select settings from development results; downstream readers contain filenames for the thesis settings. If a rerun selects different settings, align those readers with the intended run. Character offsets refer to `text_norm`; IG and Attention operate on model tokens, while word-boundary expansion for survey display is a separate step.

## LLM reference and path compatibility

The reference uses `gemini-3-flash-preview` and requires `GEMINI_API_KEY` in a root-level `.env` file for generation. It is a silver reference, not gold annotation or a training target. The main exports are `llm_test_tp_spans.csv` and `llm_test_tp_spans_argument_level.csv`.

Before reproducing the complete workflow, align these existing path differences:

- The refined MIL notebook currently sets `RUN_NAME="mil_span_run"`; evaluation and survey readers expect `mil_span_run_singles`. Use the intended run name consistently.
- The LLM notebook writes `llm_span_mask_eval_multi_mask_summary.csv`, but `methods_vs_llm.ipynb` currently reads `llm_span_mask_eval_summary.csv`.
- Some LLM cells use working-directory-relative paths and a local `/Users/karsten/...` path. Run with the repository root as the working directory and update machine-specific reads to the corresponding project path. The baseline README documents a similar local path in Random.

For the result layout and downstream consumers, see [results](../results/README.md).
