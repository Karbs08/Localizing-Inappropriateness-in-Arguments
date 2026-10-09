# Human evaluation

This directory prepares, generates, and analyzes the blinded LimeSurvey study. See the [project README](../README.md) for the completed study's findings.

## Pipeline

Run commands from the repository root with the project environment activated.

| Script | Inputs | Outputs |
|---|---|---|
| [prepare_survey_data.py](prepare_survey_data.py) | Fixed argument-level exports for Random, TF-IDF, Attention, IG, SHAP, MIL, and the LLM reference. | `survey_input/study_items.csv`, `selected_argument_ids.txt`, and `word_boundary_ratio_report.csv`. |
| [generate_limesurvey.py](generate_limesurvey.py) | Prepared study items and selected IDs. | Questionnaire import, preview, variant design, question mapping, and generation report in `survey_output/`. |
| [evaluate_survey_results.py](evaluate_survey_results.py) | LimeSurvey response CSV, matching question mapping, selected arguments, study items, and coverage report. | Analysis tables in `survey_results/` and plots in `survey_results/figures/`. |

```bash
source .venv/bin/activate
make study
```

`make study` runs preparation and questionnaire generation only. Import `survey_output/span_human_study_import.txt` into LimeSurvey and inspect the generated questionnaire before collecting new responses.

To analyze a response export with **question codes as headings and answer codes as values**:

```bash
python human_study/evaluate_survey_results.py \
  --responses human_study/survey_results/survey_results_raw.csv \
  --entity-reference-method llm
```

The same default analysis is available as `make study-evaluate`. Use `--output-dir` for a separate analysis run, and `--mapping`, `--arguments`, `--items`, and `--word-boundary-report` to supply the artifacts matching that response export. Set `--entity-reference-method none` to omit supplementary machine-reference span metrics.

## Study design

- Seven arguments from distinct issues are selected from the common test-TP pool with non-empty spans for all seven sources; the LLM reference limits the initial pool to 100 arguments.
- Four blinded explanations, **A–D**, appear per argument.
- Each source appears four times per participant, once in each display position; each source pair appears together twice.
- Seven variants rotate source assignments and argument order.
- Each explanation receives **Completeness** and **Precision** ratings on a 1–7 scale; the four explanations are then ranked.
- The completed thesis study comprises ten participants and 280 explanation evaluations, 40 per source.

Word-boundary expansion changes only the displayed highlights. Original automatic spans and perturbation scores remain unchanged. `word_boundary_ratio_report.csv` records raw and expanded coverage.

## Important artifacts

| Artifact | Purpose |
|---|---|
| `survey_output/selected_arguments.csv` | Arguments shown in the study. |
| `survey_output/variant_design.csv` | Assignment of arguments, sources, and display positions. |
| `survey_output/question_mapping.csv` | Maps blinded question codes back to explanation sources. |
| `survey_output/preview_version_1.html` | Static preview of variant 1. |
| `survey_results/human_ratings_long.csv` | Reshaped individual explanation ratings. |
| `survey_results/argument_rankings_long.csv` | Reshaped within-argument rankings. |
| `survey_results/participant_method_means.csv` | Participant-level method summaries used for analysis. |
| `survey_results/method_summary.csv` | Aggregate human scores and confidence intervals. |
| `survey_results/inter_rater_agreement.csv` | Agreement estimates for the rating criteria. |
| `survey_results/friedman_tests.csv` and `pairwise_wilcoxon_holm.csv` | Exploratory method comparisons. |

Preserve the original input, selected IDs, design, and mapping for every response export. Regenerating the survey does not recreate participant responses. The harmonic `human_f1_like` score is supplementary and is not a count-based classification F1; optional entity-span metrics use a machine reference, not human gold spans.