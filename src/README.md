# Shared Python utilities

This package provides the canonical data preparation/loading interface and shared helpers used by the experiments. Method-specific attribution, training, configuration search, and most export logic remain in the notebooks. See the [project README](../README.md) for the experimental protocol.

## Modules

| Module | Responsibility and main entry points |
|---|---|
| [prepare_data.py](prepare_data.py) | Loads or downloads the corpus, saves raw splits, creates normalized text and stable IDs, computes original classifier outputs, and writes canonical data and metadata. Run with `python -m src.prepare_data`. |
| [data.py](data.py) | Loads and validates the prepared Parquet file. Exposes `load_prepared_dataset()` and `load_prepared_splits()`. |
| [utils.py](utils.py) | Normalization, classifier inference, confusion types, span masking/deletion/highlighting, serialization, ranking helpers, AOPC exports, logit conversion, and Pareto checks. |
| [survey/limesurvey_tsv.py](survey/limesurvey_tsv.py) | `LimeSurveyTSVBuilder`, used by the active questionnaire generator. |
| [survey/survey_design.py](survey/survey_design.py) | Earlier three-explanation Fano design. The final four-explanation design is implemented in `human_study/generate_limesurvey.py`, which does not import this module. |
| [__init__.py](__init__.py) | Python package marker. Import functions directly from their defining module. |

## Prepare and load data

Run from the repository root with the project environment:

```bash
python -m src.prepare_data
# Alternatively:
make prepare-data
```

Preparation writes:

- `data/raw/appropriateness_corpus/`: the original train, validation, and test splits.
- `data/processed/appropriateness_prepared.parquet`: the shared experimental table.
- `data/processed/appropriateness_prepared_metadata.json`: source names, split sizes, dataset fingerprints, inference settings, and preparation timestamp.

```python
from src.data import load_prepared_dataset, load_prepared_splits
from src.utils import normalize_text

df_all, train_df, val_df, test_df = load_prepared_splits()
```

The loader checks required columns, unique `global_row_id` values, and the presence of all three splits. A custom prepared file can be passed through `data_file=`. Preparation metadata currently records model names, not an immutable Hugging Face model commit.

## Shared conventions

- `text_norm` uses NFKC normalization and collapsed whitespace; capitalization and punctuation are preserved.
- `LABEL_1` is the inappropriate class. `p_inappropriate_original` stores its unperturbed probability, while `predicted_score` is the confidence of the predicted class.
- Confusion types combine the corpus label with the fixed classifier prediction; MIL's bag prediction is a separate output.
- Character spans use start/exclusive-end offsets in the normalized text. Keep the text and offsets together when joining method exports.
- `mask_spans_in_text()` inserts a separate mask for each overlapping model token; `delete_spans_from_text()` removes selected character intervals. The older `apply_ablation_to_text()` uses one placeholder per character span. Use the token-aware helper for the final thesis masking protocol.
- `save_aopc_outputs()` writes the standard five ranking CSV tables to the caller-supplied directory; it does not choose the method's output path.

Data preparation uses the fixed appropriateness classifier with a maximum input length of 512 tokens and automatic CUDA/MPS/CPU selection. After changing preprocessing or prediction settings, rerun preparation and regenerate dependent outputs with compatible settings.

## Current legacy references

Some text in `data.py` still names `utils.prepare_data` or recommends `python -m scripts.prepare_data`. The actual preparation entry point in this repository is **`python -m src.prepare_data`**. Likewise, the three-source design in `survey_design.py` is not the final study's four-source design; consult the active generator and [human-study README](../human_study/README.md).
