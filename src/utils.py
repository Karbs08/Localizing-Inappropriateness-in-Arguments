"""Shared utility functions for the span-localization experiments.

This module contains project-wide helpers that are reused across the SHAP,
Integrated Gradients, attention, TF-IDF, random-baseline, and MIL notebooks.
Method-specific attribution, span-selection, training, and result-building logic
should remain in the corresponding notebook or be moved to a dedicated module.

Design principles
-----------------
- Keep shared helpers independent of notebook-global variables.
- Pass model pipelines, column names, labels, and batch sizes explicitly.
- Return copies of input data frames instead of modifying them in place.
- Keep character offsets relative to the original input text.
- Use small, composable functions with predictable return types.

Typical notebook import
-----------------------
from src.utils import (
    add_classifier_outputs,
    apply_ablation_to_text,
    attach_split,
    get_confusion_type,
    highlight_spans,
    make_json_serializable,
    mask_spans_in_text,
    normalize_text,
    predict_with_pipeline,
    trim_char_span,
)
"""

from __future__ import annotations

import html
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


DEFAULT_LABEL_COL = "Inappropriateness"
DEFAULT_PREDICTED_LABEL_COL = "predicted_label"
DEFAULT_INAPPROPRIATE_LABEL = "LABEL_1"
DEFAULT_PROBABILITY_COL = "p_inappropriate_original"
DEFAULT_SPLIT_COL = "split"

CONFUSION_FLAG_COLUMNS = {
    "TP": "is_true_positive",
    "FN": "is_false_negative",
    "FP": "is_false_positive",
    "TN": "is_true_negative",
}

__all__ = [
    "add_classifier_outputs",
    "add_classifier_predictions",
    "apply_ablation_to_text",
    "attach_split",
    "delete_spans_from_text",
    "get_confusion_type",
    "highlight_spans",
    "make_json_serializable",
    "mask_spans_in_text",
    "normalize_text",
    "predict_with_pipeline",
    "trim_char_span",
]


def attach_split(
    df: pd.DataFrame,
    split_name: str,
    *,
    split_col: str = DEFAULT_SPLIT_COL,
) -> pd.DataFrame:
    """Return a reset copy of ``df`` with the requested split label."""
    result = df.copy().reset_index(drop=True)
    result[split_col] = split_name
    return result


def normalize_text(text: Any) -> str:
    """Apply minimal Unicode and whitespace normalization.

    Capitalization and punctuation are intentionally preserved because they may
    carry relevant stylistic information for inappropriateness classification.
    Missing scalar values are converted to an empty string.
    """
    if text is None or text is pd.NA:
        return ""

    try:
        if bool(pd.isna(text)):
            return ""
    except (TypeError, ValueError):
        # Non-scalar values are converted to strings below.
        pass

    normalized = unicodedata.normalize("NFKC", str(text))
    return re.sub(r"\s+", " ", normalized).strip()


def _as_text_list(texts: str | Iterable[Any]) -> list[str]:
    """Convert a string or iterable of values to a list of strings."""
    if isinstance(texts, str):
        return [texts]
    return [str(text) for text in texts]


def _normalize_single_classifier_output(output: Any) -> list[dict[str, Any]]:
    """Normalize one pipeline result to a list of label-score dictionaries."""
    while (
        isinstance(output, list)
        and len(output) == 1
        and isinstance(output[0], list)
    ):
        output = output[0]

    if isinstance(output, Mapping):
        output = [output]

    if not isinstance(output, list) or not all(
        isinstance(item, Mapping) for item in output
    ):
        raise TypeError(
            "Unexpected classifier output. Expected a label-score dictionary "
            "or a list of label-score dictionaries."
        )

    return [dict(item) for item in output]


def _normalize_batch_classifier_outputs(
    outputs: Any,
    expected_size: int,
) -> list[list[dict[str, Any]]]:
    """Normalize Hugging Face pipeline output across transformer versions."""
    if expected_size == 1:
        if isinstance(outputs, Mapping):
            return [[dict(outputs)]]

        if isinstance(outputs, list):
            if all(isinstance(item, Mapping) for item in outputs):
                return [[dict(item) for item in outputs]]
            if len(outputs) == 1:
                return [_normalize_single_classifier_output(outputs[0])]

    if isinstance(outputs, list) and len(outputs) == expected_size:
        return [_normalize_single_classifier_output(output) for output in outputs]

    raise ValueError(
        "The classifier returned an unexpected number or structure of outputs: "
        f"expected {expected_size} result(s)."
    )


def predict_with_pipeline(
    texts: str | Iterable[Any],
    *,
    classifier: Any,
    inappropriate_label: str = DEFAULT_INAPPROPRIATE_LABEL,
    batch_size: int = 16,
    probability_col: str = DEFAULT_PROBABILITY_COL,
    progress_desc: str = "Classifier inference",
    show_progress: bool = True,
) -> pd.DataFrame:
    """Predict class probabilities and labels with a text-classification pipeline.

    The classifier is expected to return all class scores, for example from a
    Hugging Face pipeline initialized with ``top_k=None``.

    Parameters
    ----------
    texts:
        One text or an iterable of texts.
    classifier:
        Callable text-classification pipeline.
    inappropriate_label:
        Pipeline label representing the inappropriate class.
    batch_size:
        Number of texts processed per pipeline call.
    probability_col:
        Name of the returned inappropriate-probability column.
    progress_desc:
        Description shown by the progress bar.
    show_progress:
        Disable this for small examples or nested evaluations.

    Returns
    -------
    pandas.DataFrame
        One row per input text with the inappropriate probability, predicted
        label, and predicted score.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    text_list = _as_text_list(texts)
    output_columns = [probability_col, "predicted_label", "predicted_score"]

    if not text_list:
        return pd.DataFrame(columns=output_columns)

    rows: list[dict[str, Any]] = []
    starts = range(0, len(text_list), batch_size)

    for start in tqdm(
        starts,
        desc=progress_desc,
        disable=not show_progress,
    ):
        batch_texts = text_list[start : start + batch_size]
        raw_outputs = classifier(batch_texts, batch_size=batch_size)
        batch_outputs = _normalize_batch_classifier_outputs(
            raw_outputs,
            expected_size=len(batch_texts),
        )

        for output in batch_outputs:
            score_by_label: dict[str, float] = {}

            for item in output:
                if "label" not in item or "score" not in item:
                    raise KeyError(
                        "Each classifier output item must contain 'label' and 'score'."
                    )
                score_by_label[str(item["label"])] = float(item["score"])

            if not score_by_label:
                raise ValueError("The classifier returned no label scores.")

            predicted_label = max(score_by_label, key=score_by_label.get)
            rows.append(
                {
                    probability_col: score_by_label.get(
                        inappropriate_label,
                        np.nan,
                    ),
                    "predicted_label": predicted_label,
                    "predicted_score": score_by_label[predicted_label],
                }
            )

    return pd.DataFrame(rows, columns=output_columns)


def get_confusion_type(
    row: Mapping[str, Any] | pd.Series,
    *,
    label_col: str = DEFAULT_LABEL_COL,
    predicted_label_col: str = DEFAULT_PREDICTED_LABEL_COL,
    inappropriate_label: str = DEFAULT_INAPPROPRIATE_LABEL,
    positive_gold_label: int = 1,
) -> str:
    """Compute TP, FN, FP, or TN from a gold and predicted label."""
    try:
        gold_is_positive = int(row[label_col]) == positive_gold_label
    except (KeyError, TypeError, ValueError, OverflowError):
        return "UNKNOWN"

    try:
        predicted_is_positive = row[predicted_label_col] == inappropriate_label
    except KeyError:
        return "UNKNOWN"

    if gold_is_positive and predicted_is_positive:
        return "TP"
    if gold_is_positive and not predicted_is_positive:
        return "FN"
    if not gold_is_positive and predicted_is_positive:
        return "FP"
    return "TN"


def add_classifier_outputs(
    df: pd.DataFrame,
    *,
    classifier: Any,
    text_col: str = "text_norm",
    label_col: str = DEFAULT_LABEL_COL,
    inappropriate_label: str = DEFAULT_INAPPROPRIATE_LABEL,
    predicted_label_col: str = DEFAULT_PREDICTED_LABEL_COL,
    probability_col: str = DEFAULT_PROBABILITY_COL,
    batch_size: int = 16,
    show_progress: bool = True,
    overwrite: bool = True,
) -> pd.DataFrame:
    """Add classifier probabilities, labels, confusion types, and flags.

    Existing generated columns are replaced by default, which makes notebook
    cells safe to rerun without creating duplicate column names.
    """
    if text_col not in df.columns:
        raise KeyError(f"Missing text column: {text_col!r}")
    if label_col not in df.columns:
        raise KeyError(f"Missing gold-label column: {label_col!r}")

    generated_columns = [
        probability_col,
        predicted_label_col,
        "predicted_score",
        "confusion_type",
        *CONFUSION_FLAG_COLUMNS.values(),
    ]
    existing_generated = [
        column for column in generated_columns if column in df.columns
    ]

    if existing_generated and not overwrite:
        raise ValueError(
            "Generated classifier columns already exist: "
            + ", ".join(existing_generated)
        )

    result = df.drop(columns=existing_generated, errors="ignore").copy()
    result = result.reset_index(drop=True)

    prediction_df = predict_with_pipeline(
        result[text_col].astype(str).tolist(),
        classifier=classifier,
        inappropriate_label=inappropriate_label,
        batch_size=batch_size,
        probability_col=probability_col,
        show_progress=show_progress,
    )

    if predicted_label_col != DEFAULT_PREDICTED_LABEL_COL:
        prediction_df = prediction_df.rename(
            columns={DEFAULT_PREDICTED_LABEL_COL: predicted_label_col}
        )

    result = pd.concat(
        [result, prediction_df.reset_index(drop=True)],
        axis=1,
    )

    result["confusion_type"] = result.apply(
        get_confusion_type,
        axis=1,
        label_col=label_col,
        predicted_label_col=predicted_label_col,
        inappropriate_label=inappropriate_label,
    )

    for confusion_type, flag_col in CONFUSION_FLAG_COLUMNS.items():
        result[flag_col] = result["confusion_type"].eq(confusion_type)

    return result


def add_classifier_predictions(
    df: pd.DataFrame,
    **kwargs: Any,
) -> pd.DataFrame:
    """Backward-compatible alias for ``add_classifier_outputs``."""
    return add_classifier_outputs(df, **kwargs)


def trim_char_span(text: str, start: int, end: int) -> tuple[int, int]:
    """Trim leading and trailing whitespace from a character span."""
    text = str(text)
    start = max(0, int(start))
    end = min(len(text), int(end))

    if end < start:
        raise ValueError("A character span must satisfy start <= end.")

    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1

    return start, end


def _sorted_non_overlapping_spans(
    text: str,
    spans: Sequence[Mapping[str, Any]] | None,
) -> list[tuple[int, int]]:
    """Return valid, sorted, non-overlapping character offsets."""
    if not spans:
        return []

    text_length = len(text)
    prepared: list[tuple[int, int]] = []

    for span in spans:
        if "char_start" not in span or "char_end" not in span:
            raise KeyError("Each span must contain 'char_start' and 'char_end'.")

        start = max(0, int(span["char_start"]))
        end = min(text_length, int(span["char_end"]))

        if end <= start:
            continue
        prepared.append((start, end))

    prepared.sort(key=lambda item: (item[0], item[1]))

    non_overlapping: list[tuple[int, int]] = []
    last_end = 0

    for start, end in prepared:
        if start < last_end:
            continue
        non_overlapping.append((start, end))
        last_end = end

    return non_overlapping


def apply_ablation_to_text(
    text: str,
    spans: Sequence[Mapping[str, Any]] | None,
    ablation_mode: Literal["mask", "delete"],
    *,
    mask_token: str = "[MASK]",
    collapse_whitespace: bool = True,
) -> str:
    """Apply mask or delete perturbation to character spans.

    Span offsets are interpreted against the original input text. Overlapping
    spans after the first accepted span are skipped, matching the notebook logic.
    """
    text = str(text)
    offsets = _sorted_non_overlapping_spans(text, spans)

    if not offsets:
        return text
    if ablation_mode not in {"mask", "delete"}:
        raise ValueError("ablation_mode must be either 'mask' or 'delete'.")

    parts: list[str] = []
    last_end = 0

    for start, end in offsets:
        parts.append(text[last_end:start])
        if ablation_mode == "mask":
            parts.append(mask_token)
        last_end = end

    parts.append(text[last_end:])
    perturbed_text = "".join(parts)

    if collapse_whitespace:
        perturbed_text = re.sub(r"\s+", " ", perturbed_text).strip()

    return perturbed_text


def mask_spans_in_text(
    text: str,
    spans: Sequence[Mapping[str, Any]] | None,
    mask_token: str = "[MASK]",
    *,
    collapse_whitespace: bool = False,
) -> str:
    """Replace character spans with the model mask token."""
    return apply_ablation_to_text(
        text,
        spans,
        "mask",
        mask_token=mask_token,
        collapse_whitespace=collapse_whitespace,
    )


def delete_spans_from_text(
    text: str,
    spans: Sequence[Mapping[str, Any]] | None,
    *,
    collapse_whitespace: bool = True,
) -> str:
    """Delete character spans and optionally normalize remaining whitespace."""
    return apply_ablation_to_text(
        text,
        spans,
        "delete",
        collapse_whitespace=collapse_whitespace,
    )


def highlight_spans(
    text: str,
    spans: Sequence[Mapping[str, Any]] | None,
    *,
    mark_style: str = (
        "background-color:#ffe58a; padding:2px 4px; border-radius:4px;"
    ),
) -> str:
    """Return HTML with non-overlapping character spans highlighted."""
    text = str(text)
    offsets = _sorted_non_overlapping_spans(text, spans)

    if not offsets:
        return html.escape(text)

    parts: list[str] = []
    last_end = 0

    for start, end in offsets:
        parts.append(html.escape(text[last_end:start]))
        parts.append(
            f"<mark style='{html.escape(mark_style, quote=True)}'>"
            f"{html.escape(text[start:end])}</mark>"
        )
        last_end = end

    parts.append(html.escape(text[last_end:]))
    return "".join(parts)


def make_json_serializable(obj: Any) -> Any:
    """Recursively convert common NumPy and pandas values to JSON-safe types."""
    if isinstance(obj, Mapping):
        return {
            str(key): make_json_serializable(value)
            for key, value in obj.items()
        }

    if isinstance(obj, (list, tuple, set)):
        return [make_json_serializable(value) for value in obj]

    if isinstance(obj, np.ndarray):
        return [make_json_serializable(value) for value in obj.tolist()]

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        value = float(obj)
        return None if np.isnan(value) else value

    if isinstance(obj, np.bool_):
        return bool(obj)

    if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return obj.isoformat()

    if isinstance(obj, Path):
        return str(obj)

    if obj is pd.NA or obj is pd.NaT:
        return None

    try:
        if bool(pd.isna(obj)):
            return None
    except (TypeError, ValueError):
        pass

    return obj
