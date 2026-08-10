from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = REPO_ROOT / "results"
SURVEY_INPUT_DIR = REPO_ROOT / "human_study" / "survey_input"

RANDOM_SEED = 42
N_ARGUMENTS = 7


METHOD_FILES = {
    "random": RESULT_DIR / "random_baseline_results/random_best_config_test_per_argument.csv",
    "tfidf": RESULT_DIR / "tfidf_baseline_results/final_test_all_window2_topk8/tfidf_final_test_all_window2_topk8_argument_level.csv",
    "attention": RESULT_DIR / "attention_results/attention_results_word_boundary/attention_final_all_splits_last_cls_q0.6_window3_argument_level.csv",
    "ig": RESULT_DIR / "ig_results/ig_results_final/ig_final_all_splits_q0.5_window1_argument_level.csv",
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


def load_method_results():
    return {
        method: pd.read_csv(path)
        for method, path in METHOD_FILES.items()
    }


def prepare_method_data(argument_dfs):
    prepared_method_dfs = {}

    for method, original_df in argument_dfs.items():
        config = METHOD_CONFIG[method]
        df = original_df.copy()

        if "split" in df.columns:
            df = df[df["split"].astype(str).str.lower() == "test"].copy()

        if "confusion_type" in df.columns:
            df = df[
                df["confusion_type"].astype(str).str.upper() == "TP"
            ].copy()

        df["global_row_id"] = pd.to_numeric(
            df["global_row_id"],
            errors="raise",
        ).astype("Int64")

        df["post_text"] = df[config["text_col"]].astype(str)

        df["study_spans"] = df.apply(
            lambda row: build_spans(
                text=row["post_text"],
                starts=row[config["start_col"]],
                ends=row[config["end_col"]],
            ),
            axis=1,
        )

        df = df[df["study_spans"].map(len) > 0].copy()
        prepared_method_dfs[method] = df

    return prepared_method_dfs


def build_study_items(prepared_method_dfs):
    common_ids = set.intersection(
        *[
            set(df["global_row_id"])
            for df in prepared_method_dfs.values()
        ]
    )

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
                    "spans_json": json.dumps(
                        row["study_spans"],
                        ensure_ascii=False,
                    ),
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


def select_arguments(study_items):
    eligible_ids = np.array(
        sorted(study_items["global_row_id"].unique())
    )

    rng = np.random.default_rng(RANDOM_SEED)

    return rng.choice(
        eligible_ids,
        size=N_ARGUMENTS,
        replace=False,
    ).tolist()


def main():
    argument_dfs = load_method_results()
    prepared_method_dfs = prepare_method_data(argument_dfs)
    study_items = build_study_items(prepared_method_dfs)
    selected_ids = select_arguments(study_items)

    final_study_items = (
        study_items[
            study_items["global_row_id"].isin(selected_ids)
        ]
        .copy()
        .sort_values(["global_row_id", "method_id"])
    )

    SURVEY_INPUT_DIR.mkdir(parents=True, exist_ok=True)

    study_items_path = SURVEY_INPUT_DIR / "study_items.csv"
    selected_ids_path = SURVEY_INPUT_DIR / "selected_argument_ids.txt"

    final_study_items.to_csv(study_items_path, index=False)

    with selected_ids_path.open("w", encoding="utf-8") as file:
        for global_row_id in selected_ids:
            file.write(f"{global_row_id}\n")

    print(f"Wrote {study_items_path}")
    print(f"Wrote {selected_ids_path}")


if __name__ == "__main__":
    main()
