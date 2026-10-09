# Evaluation

These notebooks establish the perturbation protocol and compare fixed experimental outputs. See the [project README](../README.md) for metric definitions and thesis findings.

## Notebooks and outputs

All paths are relative to the repository root.

| Notebook | Inputs and purpose | Outputs |
|---|---|---|
| [mask_vs_delete.ipynb](mask_vs_delete.ipynb) | Uses the 88 validation true positives and generates preliminary Random, Attention, SHAP, and lightweight MIL spans for a paired operator comparison. | `results/ablation_comparison/`, with `tables/`, `plots/`, `cache/`, and `checkpoints/`. |
| [methods_vs_baselines.ipynb](methods_vs_baselines.ipynb) | Reads fixed argument-level outputs and confusion summaries for the six primary approaches; compares masking effects, coverage, statistical differences, and diagnostics. | `results/evaluation/comparison/tables/` and `plots/`. |
| [methods_vs_llm.ipynb](methods_vs_llm.ipynb) | Reads those method outputs plus the LLM reference; compares token overlap, coverage, and masking effects on shared reference arguments. | Main reference comparisons: `results/evaluation/methods_vs_llm/tables/` and `plots/`; some general outputs also go to `results/evaluation/tables/` and `plots/`. |
| [mil_document_level.ipynb](mil_document_level.ipynb) | Joins final MIL bag predictions with prepared corpus labels and original classifier predictions; evaluates document classification separately from localization. | `results/evaluation/mil/mil_classifier_evaluation_tables/` and `mil_classifier_evaluation_plots/`. |

The preliminary ablation runs **before final method configuration** and does not require final method exports or LLM annotations. The other notebooks consume fixed outputs and do not retune localization settings.

## Run order and inputs

1. Prepare the shared dataset with `make init` or `make prepare-data`.
2. Run the preliminary masking/deletion comparison. Masking is the selected operator for subsequent experiments.
3. Run the [baselines](../baselines/README.md) and [methods](../methods/README.md) to produce final exports.
4. Run the cross-method comparison. Run MIL document evaluation once final MIL predictions exist.
5. Generate or load the LLM reference and run the matched-reference comparison.

The final input filenames are listed in [results/README.md](../results/README.md). The comparison notebooks read both argument-level files and their corresponding confusion summaries; an argument-level export alone is insufficient for all cells.

The main test comparison uses **438 arguments**, with **186 true positives** as the primary localization subset. The LLM comparison starts from the **100 annotated test true positives** and checks shared IDs so that methods are evaluated on the same available arguments.

## Execution details

`methods_vs_baselines.ipynb` and `methods_vs_llm.ipynb` currently set `BASE_DIR = Path.cwd().parent`. Their kernel working directory must therefore be `evaluation/`. Starting Jupyter from the repository root does not guarantee the kernel directory; check it before running. `mask_vs_delete.ipynb` and `mil_document_level.ipynb` resolve the project root separately.

Two producer/reader differences must be aligned before a fresh full run:

- Final MIL readers expect `results/mil_results/mil_span_run_singles/`; the refined producer currently defaults to `RUN_NAME="mil_span_run"`.
- The LLM producer writes `llm_span_mask_eval_multi_mask_summary.csv`; the reference comparison reads `llm_span_mask_eval_summary.csv`.

## Interpretation

Probability drop, PDR, masked-token ratio, and class flips describe complementary aspects of the classifier response. Supplementary logit drops help inspect highly confident predictions. Overlap with LLM spans measures silver-reference agreement, not gold accuracy. MIL bag-classification scores assess a separate learned predictor and do not establish span quality by themselves.

Keep the development selection protocol separate from final test analysis, and inspect missing or duplicate IDs before interpreting aggregate comparisons.
