from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Paths and survey configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = REPO_ROOT / "results"
SURVEY_INPUT_DIR = REPO_ROOT / "human_study" / "survey_input"

RANDOM_SEED = 42
N_ARGUMENTS = 7

# Use the same tokenizer / maximum sequence length as the document-level
# appropriateness classifier. The tokenizer is used here ONLY to measure how
# much model-token coverage changes after the human-readable word-boundary
# post-processing. The classifier is NOT run again.
TOKENIZER_NAME = "timonziegenbein/appropriateness-classifier-binary"
MAX_LENGTH = 512


METHOD_FILES = {
    "random": RESULT_DIR / "random_baseline_results/random_best_config_test_per_argument.csv",
    "tfidf": RESULT_DIR / "tfidf_baseline_results/final_test_all_window2_topk8/tfidf_final_test_all_window2_topk8_argument_level.csv",
    "attention": RESULT_DIR / "attention_results/attention_results_final/attention_final_all_splits_rollout_q0.4_window0_argument_level.csv",
    "ig": RESULT_DIR / "ig_results/ig_results_final/ig_final_all_splits_q0.4_window1_argument_level.csv",
    "shap": RESULT_DIR / "shap_results/shap_results_final/shap_final_test_q0.5_window0_argument_level.csv",
    "mil": RESULT_DIR / "mil_results/mil_span_run_singles/mil_results_final/mil_final_all_splits_pool=topk_noisy_or__topk=3__spans=15__stride=4__freeze=True_argument_level.csv",
    "llm": RESULT_DIR / "llm_reference/llm_test_tp_spans_argument_level.csv",
}


METHOD_CONFIG = {
    "random": {
        "text_col": "text",
        "start_col": "merged_span_char_start_indices",
        "end_col": "merged_span_char_end_indices",
    },
    "tfidf": {
        "text_col": "text",
        "start_col": "merged_span_char_start_indices",
        "end_col": "merged_span_char_end_indices",
    },
    "attention": {
        "text_col": "text",
        "start_col": "merged_span_char_start_indices",
        "end_col": "merged_span_char_end_indices",
    },
    "ig": {
        "text_col": "text",
        "start_col": "merged_span_char_start_indices",
        "end_col": "merged_span_char_end_indices",
    },
    "shap": {
        "text_col": "text",
        "start_col": "merged_span_char_start_indices",
        "end_col": "merged_span_char_end_indices",
    },
    "mil": {
        "text_col": "text",
        "start_col": "selected_span_char_start_indices",
        "end_col": "selected_span_char_end_indices",
    },
    "llm": {
        "text_col": "argument",
        "start_col": "llm_char_starts",
        "end_col": "llm_char_ends",
    },
}


# ---------------------------------------------------------------------------
# Parsing and span construction
# ---------------------------------------------------------------------------

def parse_list(value):
    if value is None:
        return []

    if isinstance(value, (list, tuple, np.ndarray)):
        return list(value)

    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass

    if isinstance(value, str):
        value = value.strip()

        if not value:
            return []

        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

        parsed = ast.literal_eval(value)
        if isinstance(parsed, (list, tuple)):
            return list(parsed)

    raise ValueError(f"Could not parse list value: {value!r}")


def merge_intervals(intervals):
    if not intervals:
        return []

    intervals = sorted(intervals)
    merged = [list(intervals[0])]

    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    return [tuple(interval) for interval in merged]


def build_spans(text, starts, ends):
    """
    Build the raw character spans exactly from the final method result file.
    """
    text = str(text)
    starts = parse_list(starts)
    ends = parse_list(ends)

    intervals = merge_intervals(
        [(int(start), int(end)) for start, end in zip(starts, ends)]
    )

    return [
        {
            "start": start,
            "end": end,
            "text": text[start:end],
        }
        for start, end in intervals
    ]


# ---------------------------------------------------------------------------
# Human-readable word-boundary post-processing
# ---------------------------------------------------------------------------

def get_word_spans(text):
    """
    Return word-like character spans.

    Common contractions and hyphenated words are kept as one word, e.g.
    "can't", "don't", and "day-to-day".
    """
    text = str(text)
    pattern = re.compile(r"\b\w+(?:['’\-]\w+)*\b", flags=re.UNICODE)

    return [
        {
            "start": int(match.start()),
            "end": int(match.end()),
            "text": match.group(0),
        }
        for match in pattern.finditer(text)
    ]


def expand_span_to_word_boundaries(span, word_spans):
    """
    Normalize one span to complete boundaries of all words it already overlaps.

    The operation does not add unrelated neighbouring words. It only completes
    words that are already touched by the original character span.

    A punctuation-only span has no overlapping word and returns None.
    """
    span_start = int(span["start"])
    span_end = int(span["end"])

    overlapping_words = [
        word
        for word in word_spans
        if word["start"] < span_end and word["end"] > span_start
    ]

    if not overlapping_words:
        return None

    return {
        "start": min(word["start"] for word in overlapping_words),
        "end": max(word["end"] for word in overlapping_words),
    }


def apply_word_boundary_postprocessing(
    text,
    raw_spans,
    *,
    drop_punctuation_only=False,
):
    """
    Convert extracted spans to human-readable full-word spans.

    Steps:
    1. expand/normalize every raw span to the complete word(s) it overlaps,
    2. optionally drop punctuation-only spans,
    3. merge spans that overlap/touch after normalization,
    4. fall back to the raw spans if all spans would otherwise disappear.

    The same function is applied to every method. Methods that already return
    complete words therefore remain unchanged in practice.
    """
    text = str(text)
    word_spans = get_word_spans(text)

    boundary_intervals = []

    for span in raw_spans:
        expanded = expand_span_to_word_boundaries(span, word_spans)

        if expanded is None:
            if not drop_punctuation_only:
                boundary_intervals.append(
                    (int(span["start"]), int(span["end"]))
                )
            continue

        boundary_intervals.append(
            (int(expanded["start"]), int(expanded["end"]))
        )

    boundary_intervals = merge_intervals(boundary_intervals)

    # Fallback from the supplied attention post-processing logic:
    # do not accidentally turn a non-empty explanation into an empty one.
    fallback_used = False
    if not boundary_intervals and raw_spans:
        boundary_intervals = [
            (int(span["start"]), int(span["end"]))
            for span in raw_spans
        ]
        boundary_intervals = merge_intervals(boundary_intervals)
        fallback_used = True

    processed_spans = [
        {
            "start": start,
            "end": end,
            "text": text[start:end],
        }
        for start, end in boundary_intervals
    ]

    raw_intervals = [
        (int(span["start"]), int(span["end"]))
        for span in raw_spans
    ]
    processed_intervals = [
        (int(span["start"]), int(span["end"]))
        for span in processed_spans
    ]

    return processed_spans, fallback_used, raw_intervals != processed_intervals


# ---------------------------------------------------------------------------
# Token-coverage control metric
# ---------------------------------------------------------------------------

def get_model_token_offsets(text, tokenizer):
    """
    Return non-special model-token offsets for the classifier tokenizer.

    This is used only to quantify the change in highlighted token coverage.
    No masking and no classifier inference is performed here.
    """
    encoded = tokenizer(
        str(text),
        truncation=True,
        max_length=MAX_LENGTH,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
        add_special_tokens=True,
    )

    token_rows = []

    for token_idx, ((start, end), is_special) in enumerate(
        zip(
            encoded["offset_mapping"],
            encoded["special_tokens_mask"],
        )
    ):
        start = int(start)
        end = int(end)

        if int(is_special) == 0 and end > start:
            token_rows.append(
                {
                    "token_idx": int(token_idx),
                    "start": start,
                    "end": end,
                }
            )

    return token_rows


def get_span_token_coverage(spans, token_rows):
    """
    Count unique model tokens that overlap at least one supplied span.
    """
    selected_token_indices = set()

    for span in spans:
        span_start = int(span["start"])
        span_end = int(span["end"])

        for token in token_rows:
            if token["start"] < span_end and token["end"] > span_start:
                selected_token_indices.add(token["token_idx"])

    n_model_tokens = len(token_rows)
    n_selected_tokens = len(selected_token_indices)

    token_ratio = (
        n_selected_tokens / n_model_tokens
        if n_model_tokens > 0
        else np.nan
    )

    return n_model_tokens, n_selected_tokens, token_ratio


def add_word_boundary_metadata(df, tokenizer):
    """
    Apply the survey-only word-boundary normalization and add coverage metadata.
    """
    processed_rows = []

    for _, row in df.iterrows():
        row = row.copy()
        text = str(row["post_text"])
        raw_spans = row["study_spans_raw"]

        processed_spans, fallback_used, changed = (
            apply_word_boundary_postprocessing(
                text=text,
                raw_spans=raw_spans,
                drop_punctuation_only=True,
            )
        )

        token_rows = get_model_token_offsets(text, tokenizer)

        (
            n_model_tokens,
            n_selected_tokens_raw,
            token_ratio_raw,
        ) = get_span_token_coverage(raw_spans, token_rows)

        (
            _,
            n_selected_tokens_post,
            token_ratio_post,
        ) = get_span_token_coverage(processed_spans, token_rows)

        row["study_spans"] = processed_spans

        row["word_boundary_changed"] = bool(changed)
        row["word_boundary_fallback_used"] = bool(fallback_used)

        row["survey_n_model_tokens"] = int(n_model_tokens)
        row["survey_n_selected_tokens_raw"] = int(n_selected_tokens_raw)
        row["survey_n_selected_tokens_postprocessed"] = int(
            n_selected_tokens_post
        )

        row["survey_masked_token_ratio_raw"] = float(token_ratio_raw)
        row["survey_masked_token_ratio_postprocessed"] = float(
            token_ratio_post
        )
        row["survey_masked_token_ratio_delta"] = float(
            token_ratio_post - token_ratio_raw
        )

        processed_rows.append(row)

    return pd.DataFrame(processed_rows)


# ---------------------------------------------------------------------------
# Load and prepare final method outputs
# ---------------------------------------------------------------------------

def load_method_results():
    return {
        method: pd.read_csv(path)
        for method, path in METHOD_FILES.items()
    }


def prepare_method_data(argument_dfs, tokenizer):
    prepared_method_dfs = {}

    for method, original_df in argument_dfs.items():
        config = METHOD_CONFIG[method]
        df = original_df.copy()

        if "split" in df.columns:
            df = df[
                df["split"].astype(str).str.lower() == "test"
            ].copy()

        if "confusion_type" in df.columns:
            df = df[
                df["confusion_type"].astype(str).str.upper() == "TP"
            ].copy()

        df["global_row_id"] = pd.to_numeric(
            df["global_row_id"],
            errors="raise",
        ).astype("Int64")

        df["post_text"] = df[config["text_col"]].astype(str)

        # Raw final spans as produced by the method/configuration.
        df["study_spans_raw"] = df.apply(
            lambda row: build_spans(
                text=row["post_text"],
                starts=row[config["start_col"]],
                ends=row[config["end_col"]],
            ),
            axis=1,
        )

        df = df[df["study_spans_raw"].map(len) > 0].copy()

        # Central survey-only post-processing for all seven methods.
        df = add_word_boundary_metadata(df, tokenizer)

        prepared_method_dfs[method] = df

    return prepared_method_dfs


# ---------------------------------------------------------------------------
# Build common survey pool
# ---------------------------------------------------------------------------

def build_study_pool(prepared_method_dfs):
    # The 100 LLM-annotated test TPs define the candidate pool. An argument is
    # retained only if a final output exists for every evaluated method.
    common_ids = set(prepared_method_dfs["llm"]["global_row_id"])

    for method, df in prepared_method_dfs.items():
        if method == "llm":
            continue
        common_ids &= set(df["global_row_id"])

    study_rows = []

    for method, df in prepared_method_dfs.items():
        subset = df[df["global_row_id"].isin(common_ids)].copy()

        for _, row in subset.iterrows():
            study_rows.append(
                {
                    "global_row_id": int(row["global_row_id"]),
                    "method_id": method,
                    "post_id": row.get("post_id", None),
                    "issue": row["issue"],
                    "post_text": row["post_text"],

                    # The survey generator reads spans_json. These are now the
                    # post-processed, human-readable spans.
                    "spans_json": json.dumps(
                        row["study_spans"],
                        ensure_ascii=False,
                    ),

                    # Raw spans are retained for transparency / later analysis.
                    "raw_spans_json": json.dumps(
                        row["study_spans_raw"],
                        ensure_ascii=False,
                    ),

                    "word_boundary_changed": bool(
                        row["word_boundary_changed"]
                    ),
                    "word_boundary_fallback_used": bool(
                        row["word_boundary_fallback_used"]
                    ),

                    # Coverage control values. "masked" is retained in the
                    # name because it matches the existing thesis metric; no
                    # new masking or classifier inference is performed here.
                    "survey_n_model_tokens": row[
                        "survey_n_model_tokens"
                    ],
                    "survey_n_selected_tokens_raw": row[
                        "survey_n_selected_tokens_raw"
                    ],
                    "survey_n_selected_tokens_postprocessed": row[
                        "survey_n_selected_tokens_postprocessed"
                    ],
                    "survey_masked_token_ratio_raw": row[
                        "survey_masked_token_ratio_raw"
                    ],
                    "survey_masked_token_ratio_postprocessed": row[
                        "survey_masked_token_ratio_postprocessed"
                    ],
                    "survey_masked_token_ratio_delta": row[
                        "survey_masked_token_ratio_delta"
                    ],

                    # Existing automatic-evaluation metadata is kept exactly
                    # as produced by the original method run.
                    "prob_drop": row.get("prob_drop", np.nan),
                    "masked_token_ratio": row.get(
                        "masked_token_ratio",
                        np.nan,
                    ),
                    "p_inappropriate_original": row.get(
                        "p_inappropriate_original",
                        np.nan,
                    ),
                    "p_inappropriate_masked": row.get(
                        "p_inappropriate_masked",
                        np.nan,
                    ),
                }
            )

    return (
        pd.DataFrame(study_rows)
        .sort_values(["global_row_id", "method_id"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Sampling and word-boundary reporting
# ---------------------------------------------------------------------------

def select_arguments(study_pool):
    # One row per eligible argument.
    candidates = (
        study_pool[["global_row_id", "issue"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    unique_issues = candidates["issue"].dropna().unique()

    if len(unique_issues) < N_ARGUMENTS:
        raise ValueError(
            f"Need at least {N_ARGUMENTS} unique issues, "
            f"but only {len(unique_issues)} are available."
        )

    rng = np.random.default_rng(RANDOM_SEED)

    # First sample N distinct issues.
    selected_issues = rng.choice(
        unique_issues,
        size=N_ARGUMENTS,
        replace=False,
    )

    # Then sample exactly one argument from each selected issue.
    selected_ids = []

    for issue in selected_issues:
        issue_ids = (
            candidates.loc[
                candidates["issue"] == issue,
                "global_row_id",
            ]
            .to_numpy()
        )

        selected_id = rng.choice(issue_ids)
        selected_ids.append(int(selected_id))

    return selected_ids


def summarize_word_boundary_effect(df, scope):
    summary = (
        df.groupby("method_id", as_index=False)
        .agg(
            n_arguments=("global_row_id", "nunique"),
            n_changed=("word_boundary_changed", "sum"),
            mean_masked_token_ratio_raw=(
                "survey_masked_token_ratio_raw",
                "mean",
            ),
            mean_masked_token_ratio_postprocessed=(
                "survey_masked_token_ratio_postprocessed",
                "mean",
            ),
            mean_masked_token_ratio_delta=(
                "survey_masked_token_ratio_delta",
                "mean",
            ),
        )
    )

    summary["changed_share"] = (
        summary["n_changed"] / summary["n_arguments"]
    )
    summary.insert(0, "scope", scope)

    return summary


def print_word_boundary_summary(summary):
    print()
    print("WORD-BOUNDARY TOKEN-COVERAGE CONTROL")
    print("(coverage only; no classifier re-scoring)")
    print("-" * 72)

    cols = [
        "scope",
        "method_id",
        "n_arguments",
        "n_changed",
        "mean_masked_token_ratio_raw",
        "mean_masked_token_ratio_postprocessed",
        "mean_masked_token_ratio_delta",
    ]

    print(
        summary[cols].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Loading tokenizer: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_NAME,
        use_fast=True,
    )

    argument_dfs = load_method_results()
    prepared_method_dfs = prepare_method_data(
        argument_dfs,
        tokenizer,
    )

    study_pool = build_study_pool(prepared_method_dfs)
    selected_ids = select_arguments(study_pool)

    final_study_items = (
        study_pool[
            study_pool["global_row_id"].isin(selected_ids)
        ]
        .copy()
        .sort_values(["global_row_id", "method_id"])
        .reset_index(drop=True)
    )

    SURVEY_INPUT_DIR.mkdir(parents=True, exist_ok=True)

    study_items_path = (
        SURVEY_INPUT_DIR / "study_items.csv"
    )
    selected_ids_path = (
        SURVEY_INPUT_DIR / "selected_argument_ids.txt"
    )
    ratio_report_path = (
        SURVEY_INPUT_DIR / "word_boundary_ratio_report.csv"
    )

    final_study_items.to_csv(
        study_items_path,
        index=False,
    )

    with selected_ids_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for global_row_id in selected_ids:
            file.write(f"{global_row_id}\n")

    pool_summary = summarize_word_boundary_effect(
        study_pool,
        scope="eligible_pool",
    )
    selected_summary = summarize_word_boundary_effect(
        final_study_items,
        scope="selected_survey_arguments",
    )

    ratio_report = pd.concat(
        [pool_summary, selected_summary],
        ignore_index=True,
    )

    ratio_report.to_csv(
        ratio_report_path,
        index=False,
    )

    print(
        f"Eligible LLM-based survey pool: "
        f"{study_pool['global_row_id'].nunique()} arguments"
    )
    print(
        f"Selected {len(selected_ids)} arguments "
        f"with seed {RANDOM_SEED}"
    )

    print_word_boundary_summary(ratio_report)

    print()
    print(f"Wrote {study_items_path}")
    print(f"Wrote {selected_ids_path}")
    print(f"Wrote {ratio_report_path}")


if __name__ == "__main__":
    main()
