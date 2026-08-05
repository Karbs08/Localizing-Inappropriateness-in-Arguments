"""Shared source package for the span-localization experiments."""

from .utils import (
    add_classifier_outputs,
    add_classifier_predictions,
    apply_ablation_to_text,
    attach_split,
    delete_spans_from_text,
    get_confusion_type,
    highlight_spans,
    make_json_serializable,
    mask_spans_in_text,
    normalize_text,
    predict_with_pipeline,
    trim_char_span,
)

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
