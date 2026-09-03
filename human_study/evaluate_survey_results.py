from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import krippendorff


REPO_ROOT = Path(__file__).resolve().parents[1]
HUMAN_STUDY_DIR = REPO_ROOT / "human_study"
DEFAULT_MAPPING = HUMAN_STUDY_DIR / "survey_output" / "question_mapping.csv"
DEFAULT_ARGUMENTS = HUMAN_STUDY_DIR / "survey_output" / "selected_arguments.csv"
DEFAULT_ITEMS = HUMAN_STUDY_DIR / "survey_input" / "study_items.csv"
DEFAULT_OUTPUT = HUMAN_STUDY_DIR / "survey_results"

RATING_SCALE_MIN = 1
RATING_SCALE_MAX = 7
DISPLAY_LABELS = ("A", "B", "C", "D")
CRITERIA = ("COM", "PRE")

METHOD_ORDER = [
    "random",
    "tfidf",
    "attention",
    "ig",
    "shap",
    "mil",
    "llm",
]

METHOD_LABELS = {
    "random": "Random",
    "tfidf": "TF-IDF",
    "attention": "Attention",
    "ig": "Integrated Gradients",
    "shap": "SHAP",
    "mil": "MIL",
    "llm": "LLM",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reshape and analyze LimeSurvey responses for the human span study. "
            "Export LimeSurvey responses using question codes and answer codes."
        )
    )
    parser.add_argument(
        "--responses",
        type=Path,
        required=True,
        help="Raw CSV export from LimeSurvey using question codes as headings.",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=DEFAULT_MAPPING,
    )
    parser.add_argument(
        "--arguments",
        type=Path,
        default=DEFAULT_ARGUMENTS,
    )
    parser.add_argument(
        "--items",
        type=Path,
        default=DEFAULT_ITEMS,
        help="study_items.csv used to generate the survey.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--entity-reference-method",
        default="llm",
        help=(
            "Optional method whose spans are used as a reference for supplementary "
            "NER-style exact/partial span metrics. Use 'none' to disable."
        ),
    )
    return parser.parse_args()


def normalized_code(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", str(value)).upper()


def build_column_lookup(columns) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for col in columns:
        key = normalized_code(col)
        if key and key not in lookup:
            lookup[key] = col
    return lookup


def find_column(lookup: dict[str, str], *candidates: str) -> str | None:
    for candidate in candidates:
        key = normalized_code(candidate)
        if key in lookup:
            return lookup[key]
    return None


def parse_rating(value):
    if pd.isna(value):
        return pd.NA

    text = str(value).strip()
    if not text:
        return pd.NA

    match = re.match(r"^\s*(\d+)", text)
    if match:
        return int(match.group(1))

    return pd.NA


def parse_ranked_label(value):
    if pd.isna(value):
        return pd.NA

    text = str(value).strip()
    if not text:
        return pd.NA

    upper = text.upper()
    if upper in DISPLAY_LABELS:
        return upper

    match = re.search(r"EXPLANATION\s+([ABCD])", upper)
    if match:
        return match.group(1)

    return pd.NA


def normalize_likert(series: pd.Series) -> pd.Series:
    return (series - RATING_SCALE_MIN) / (
        RATING_SCALE_MAX - RATING_SCALE_MIN
    )


def harmonic_quality(precision_norm, completeness_norm):
    if pd.isna(precision_norm) or pd.isna(completeness_norm):
        return np.nan

    denominator = precision_norm + completeness_norm
    if denominator == 0:
        return 0.0

    return 2 * precision_norm * completeness_norm / denominator


def reshape_responses(
    responses: pd.DataFrame,
    mapping: pd.DataFrame,
    arguments: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lookup = build_column_lookup(responses.columns)

    version_col = find_column(lookup, "VERSION")
    if version_col is None:
        raise KeyError(
            "Could not find VERSION. Export LimeSurvey responses with "
            "'Question code' headings and enable ExpressionScript codes."
        )

    response_id_col = next(
        (
            col
            for col in responses.columns
            if normalized_code(col) in {"RESPONSEID", "ID"}
        ),
        None,
    )
    seed_col = next(
        (
            col
            for col in responses.columns
            if normalized_code(col) == "SEED"
        ),
        None,
    )

    mapping = mapping.copy()
    mapping["version"] = mapping["version"].astype(int)
    mapping["global_row_id"] = mapping["global_row_id"].astype(str)

    rating_rows: list[dict] = []
    ranking_rows: list[dict] = []

    for response_index, response in responses.iterrows():
        version_value = response[version_col]
        if pd.isna(version_value) or str(version_value).strip() == "":
            continue

        version = int(float(version_value))
        response_id = (
            response[response_id_col]
            if response_id_col is not None
            else response_index + 1
        )
        lime_seed = response[seed_col] if seed_col is not None else pd.NA

        participant_mapping = (
            mapping[mapping["version"] == version]
            .sort_values(["display_position", "display_label"])
            .copy()
        )

        for (
            global_row_id,
            display_position,
            argument_index,
        ), group in participant_mapping.groupby(
            ["global_row_id", "display_position", "argument_index"],
            sort=False,
        ):
            group = group.sort_values("display_label")
            first = group.iloc[0]
            ranking_code = str(first["ranking_question_code"])
            comment_code = str(first["comment_question_code"])

            label_to_method = dict(
                zip(group["display_label"], group["method_id"])
            )

            # Ranking questions are exported as QCODE_1, QCODE_2, ... where
            # each column stores the answer code ranked at that position.
            ranking_by_label: dict[str, int] = {}
            for rank_position in range(1, len(DISPLAY_LABELS) + 1):
                rank_col = find_column(
                    lookup,
                    f"{ranking_code}_{rank_position}",
                    f"{ranking_code}{rank_position}",
                )
                if rank_col is None:
                    continue

                label = parse_ranked_label(response[rank_col])
                if pd.notna(label):
                    ranking_by_label[str(label)] = rank_position
                    ranking_rows.append(
                        {
                            "response_id": response_id,
                            "version": version,
                            "lime_seed": lime_seed,
                            "display_position": int(display_position),
                            "argument_index": int(argument_index),
                            "global_row_id": str(global_row_id),
                            "rank": rank_position,
                            "display_label": str(label),
                            "method_id": str(label_to_method[str(label)]),
                        }
                    )

            comment_col = find_column(lookup, comment_code)
            comment = (
                response[comment_col]
                if comment_col is not None and pd.notna(response[comment_col])
                else ""
            )

            for _, item in group.iterrows():
                rating_code = str(item["rating_question_code"])
                criterion_values = {}

                for criterion in CRITERIA:
                    criterion_col = find_column(
                        lookup,
                        f"{rating_code}{criterion}",
                        f"{rating_code}_{criterion}",
                        f"{rating_code}[{criterion}]",
                        f"{rating_code}.{criterion}",
                    )
                    criterion_values[criterion] = (
                        parse_rating(response[criterion_col])
                        if criterion_col is not None
                        else pd.NA
                    )

                display_label = str(item["display_label"])
                rank = ranking_by_label.get(display_label, pd.NA)

                rating_rows.append(
                    {
                        "response_id": response_id,
                        "version": version,
                        "lime_seed": lime_seed,
                        "display_position": int(display_position),
                        "argument_index": int(argument_index),
                        "global_row_id": str(global_row_id),
                        "method_id": str(item["method_id"]),
                        "display_label": display_label,
                        "completeness": criterion_values["COM"],
                        "precision": criterion_values["PRE"],
                        "rank": rank,
                        "comment": comment,
                    }
                )

    ratings = pd.DataFrame(rating_rows)
    rankings = pd.DataFrame(ranking_rows)

    for col in ("completeness", "precision", "rank"):
        if col in ratings.columns:
            ratings[col] = pd.to_numeric(ratings[col], errors="coerce")

    if not ratings.empty:
        ratings["completeness_norm"] = normalize_likert(
            ratings["completeness"]
        )
        ratings["precision_norm"] = normalize_likert(
            ratings["precision"]
        )
        ratings["human_f1_like"] = ratings.apply(
            lambda row: harmonic_quality(
                row["precision_norm"],
                row["completeness_norm"],
            ),
            axis=1,
        )
        # Optional 1-7 projection for figures that should use the same visual
        # scale as the original ratings.
        ratings["human_f1_like_1_7"] = (
            RATING_SCALE_MIN
            + (RATING_SCALE_MAX - RATING_SCALE_MIN)
            * ratings["human_f1_like"]
        )
        ratings["rank_score"] = (
            len(DISPLAY_LABELS) - ratings["rank"]
        ) / (len(DISPLAY_LABELS) - 1)
        ratings["is_first_rank"] = ratings["rank"].eq(1)

    if arguments is not None and not ratings.empty:
        arguments = arguments.copy()
        arguments["global_row_id"] = arguments["global_row_id"].astype(str)
        extra_cols = [
            col
            for col in ("global_row_id", "issue", "post_text")
            if col in arguments.columns
        ]
        ratings = ratings.merge(
            arguments[extra_cols].drop_duplicates("global_row_id"),
            on="global_row_id",
            how="left",
        )
        if not rankings.empty:
            rankings = rankings.merge(
                arguments[extra_cols].drop_duplicates("global_row_id"),
                on="global_row_id",
                how="left",
            )

    return ratings, rankings


def compute_inter_rater_agreement(
    ratings: pd.DataFrame,
    output_dir: Path,
):
    agreement_rows = []

    for metric in ("completeness", "precision"):
        data = ratings[
            [
                "response_id",
                "global_row_id",
                "method_id",
                metric,
            ]
        ].dropna(subset=[metric]).copy()

        # One unit corresponds to one concrete explanation:
        # argument × localization method.
        data["unit_id"] = (
            data["global_row_id"].astype(str)
            + "__"
            + data["method_id"].astype(str)
        )

        if data.duplicated(["response_id", "unit_id"]).any():
            raise ValueError(
                f"Duplicate ratings found for {metric}."
            )

        # Rows = participants, columns = explanation items.
        matrix = data.pivot(
            index="response_id",
            columns="unit_id",
            values=metric,
        )

        # Units rated by only one participant cannot contribute
        # to inter-rater agreement.
        ratings_per_unit = matrix.notna().sum(axis=0)
        matrix = matrix.loc[:, ratings_per_unit >= 2]

        alpha = krippendorff.alpha(
            reliability_data=matrix.to_numpy(dtype=float),
            level_of_measurement="ordinal",
            value_domain=np.arange(
                RATING_SCALE_MIN,
                RATING_SCALE_MAX + 1,
            ),
        )

        agreement_rows.append(
            {
                "metric": metric,
                "krippendorff_alpha_ordinal": alpha,
                "n_participants": matrix.shape[0],
                "n_units": matrix.shape[1],
                "n_ratings": int(matrix.notna().sum().sum()),
                "mean_raters_per_unit": (
                    matrix.notna()
                    .sum(axis=0)
                    .mean()
                ),
                "min_raters_per_unit": int(
                    matrix.notna()
                    .sum(axis=0)
                    .min()
                ),
                "max_raters_per_unit": int(
                    matrix.notna()
                    .sum(axis=0)
                    .max()
                ),
            }
        )

    agreement = pd.DataFrame(agreement_rows)

    return agreement


def mean_ci(values, confidence=0.95):
    values = pd.Series(values).dropna().astype(float)
    n = len(values)

    if n == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
        }

    mean = values.mean()
    median = values.median()
    std = values.std(ddof=1) if n > 1 else np.nan

    if n > 1:
        sem = stats.sem(values)
        margin = stats.t.ppf((1 + confidence) / 2, n - 1) * sem
        ci_low = mean - margin
        ci_high = mean + margin
    else:
        ci_low = np.nan
        ci_high = np.nan

    return {
        "n": n,
        "mean": mean,
        "median": median,
        "std": std,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    m = len(p_values)
    adjusted = np.empty(m, dtype=float)
    running_max = 0.0

    for rank, index in enumerate(order):
        candidate = (m - rank) * p_values[index]
        running_max = max(running_max, candidate)
        adjusted[index] = min(running_max, 1.0)

    return adjusted


def build_method_summaries(ratings: pd.DataFrame, output_dir: Path):
    metrics = (
        "completeness",
        "precision",
        "human_f1_like",
        "human_f1_like_1_7",
        "rank",
        "rank_score",
        "is_first_rank",
    )

    # Primary summaries use one mean per participant and method. This prevents
    # multiple argument ratings by the same participant from being treated as
    # independent observations.
    participant_method = (
        ratings.groupby(["response_id", "method_id"], as_index=False)[
            list(metrics)
        ]
        .mean()
    )
    participant_method.to_csv(
        output_dir / "participant_method_means.csv",
        index=False,
    )

    summary_rows = []
    for method in METHOD_ORDER:
        method_df = participant_method[
            participant_method["method_id"] == method
        ]
        for metric in metrics:
            result = mean_ci(method_df[metric])
            summary_rows.append(
                {
                    "method_id": method,
                    "method": METHOD_LABELS.get(method, method),
                    "metric": metric,
                    **result,
                }
            )

    method_summary = pd.DataFrame(summary_rows)
    method_summary.to_csv(
        output_dir / "method_summary.csv",
        index=False,
    )

    return participant_method, method_summary


def build_pairwise_ranking(ratings: pd.DataFrame, output_dir: Path):
    pair_rows = []

    for (
        response_id,
        global_row_id,
    ), group in ratings.dropna(subset=["rank"]).groupby(
        ["response_id", "global_row_id"]
    ):
        rows = list(group.itertuples(index=False))
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a = rows[i]
                b = rows[j]
                if a.rank == b.rank:
                    continue

                winner = a.method_id if a.rank < b.rank else b.method_id
                loser = b.method_id if a.rank < b.rank else a.method_id
                pair_rows.append(
                    {
                        "response_id": response_id,
                        "global_row_id": global_row_id,
                        "winner": winner,
                        "loser": loser,
                    }
                )

    pairwise = pd.DataFrame(pair_rows)
    pairwise.to_csv(
        output_dir / "pairwise_ranking_outcomes.csv",
        index=False,
    )

    matrix = pd.DataFrame(
        np.nan,
        index=METHOD_ORDER,
        columns=METHOD_ORDER,
    )
    long_rows = []

    for method_a in METHOD_ORDER:
        for method_b in METHOD_ORDER:
            if method_a == method_b:
                continue

            wins = len(
                pairwise[
                    (pairwise["winner"] == method_a)
                    & (pairwise["loser"] == method_b)
                ]
            )
            losses = len(
                pairwise[
                    (pairwise["winner"] == method_b)
                    & (pairwise["loser"] == method_a)
                ]
            )
            total = wins + losses
            win_rate = wins / total if total else np.nan
            matrix.loc[method_a, method_b] = win_rate
            long_rows.append(
                {
                    "method_a": method_a,
                    "method_b": method_b,
                    "wins_a": wins,
                    "wins_b": losses,
                    "comparisons": total,
                    "win_rate_a": win_rate,
                }
            )

    pd.DataFrame(long_rows).to_csv(
        output_dir / "pairwise_ranking_summary.csv",
        index=False,
    )
    matrix.to_csv(output_dir / "pairwise_win_rate_matrix.csv")

    return pairwise, matrix


def exploratory_repeated_measures(
    participant_method: pd.DataFrame,
    output_dir: Path,
):
    metrics = ("completeness", "precision", "human_f1_like")

    friedman_rows = []
    pairwise_rows = []

    for metric in metrics:
        pivot = (
            participant_method
            .pivot(index="response_id", columns="method_id", values=metric)
            .reindex(columns=METHOD_ORDER)
        )

        complete = pivot.dropna()
        if len(complete) >= 2:
            result = stats.friedmanchisquare(
                *[complete[method].values for method in METHOD_ORDER]
            )
            friedman_rows.append(
                {
                    "metric": metric,
                    "n_participants": len(complete),
                    "friedman_chi2": result.statistic,
                    "p_value": result.pvalue,
                }
            )
        else:
            friedman_rows.append(
                {
                    "metric": metric,
                    "n_participants": len(complete),
                    "friedman_chi2": np.nan,
                    "p_value": np.nan,
                }
            )

        metric_pairs = []
        for i, method_a in enumerate(METHOD_ORDER):
            for method_b in METHOD_ORDER[i + 1:]:
                pair = pivot[[method_a, method_b]].dropna()
                if len(pair) < 2:
                    statistic = np.nan
                    p_value = np.nan
                else:
                    diff = pair[method_a] - pair[method_b]
                    if np.allclose(diff, 0):
                        statistic = 0.0
                        p_value = 1.0
                    else:
                        result = stats.wilcoxon(
                            pair[method_a],
                            pair[method_b],
                            alternative="two-sided",
                        )
                        statistic = result.statistic
                        p_value = result.pvalue

                metric_pairs.append(
                    {
                        "metric": metric,
                        "method_a": method_a,
                        "method_b": method_b,
                        "n": len(pair),
                        "mean_a": pair[method_a].mean() if len(pair) else np.nan,
                        "mean_b": pair[method_b].mean() if len(pair) else np.nan,
                        "mean_difference": (
                            (pair[method_a] - pair[method_b]).mean()
                            if len(pair)
                            else np.nan
                        ),
                        "wilcoxon_statistic": statistic,
                        "p_value": p_value,
                    }
                )

        valid_indices = [
            i
            for i, row in enumerate(metric_pairs)
            if pd.notna(row["p_value"])
        ]
        if valid_indices:
            adjusted = holm_adjust(
                [metric_pairs[i]["p_value"] for i in valid_indices]
            )
            for idx, adj in zip(valid_indices, adjusted):
                metric_pairs[idx]["p_holm"] = adj
        for row in metric_pairs:
            row.setdefault("p_holm", np.nan)

        pairwise_rows.extend(metric_pairs)

    pd.DataFrame(friedman_rows).to_csv(
        output_dir / "friedman_tests.csv",
        index=False,
    )
    pd.DataFrame(pairwise_rows).to_csv(
        output_dir / "pairwise_wilcoxon_holm.csv",
        index=False,
    )


def parse_spans_json(value) -> list[tuple[int, int]]:
    spans = json.loads(value) if isinstance(value, str) else value
    return [
        (int(span["start"]), int(span["end"]))
        for span in spans
    ]


def span_iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    intersection = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    if intersection == 0:
        return 0.0
    union = max(a[1], b[1]) - min(a[0], b[0])
    return intersection / union if union else 0.0


def ner_style_span_counts(
    predicted: list[tuple[int, int]],
    reference: list[tuple[int, int]],
):
    pred_remaining = list(range(len(predicted)))
    ref_remaining = list(range(len(reference)))

    exact = 0
    matched_ious = []

    # Exact matches first.
    for pred_idx in list(pred_remaining):
        match_ref = next(
            (
                ref_idx
                for ref_idx in ref_remaining
                if predicted[pred_idx] == reference[ref_idx]
            ),
            None,
        )
        if match_ref is not None:
            exact += 1
            matched_ious.append(1.0)
            pred_remaining.remove(pred_idx)
            ref_remaining.remove(match_ref)

    # Then match remaining spans one-to-one by highest positive IoU.
    overlap_candidates = []
    for pred_idx in pred_remaining:
        for ref_idx in ref_remaining:
            iou = span_iou(predicted[pred_idx], reference[ref_idx])
            if iou > 0:
                overlap_candidates.append((iou, pred_idx, ref_idx))

    overlap_candidates.sort(reverse=True)
    used_pred = set()
    used_ref = set()
    partial = 0

    for iou, pred_idx, ref_idx in overlap_candidates:
        if pred_idx in used_pred or ref_idx in used_ref:
            continue
        used_pred.add(pred_idx)
        used_ref.add(ref_idx)
        partial += 1
        matched_ious.append(iou)

    matched = exact + partial

    return {
        "exact": exact,
        "partial": partial,
        "predicted": len(predicted),
        "reference": len(reference),
        "missed": len(reference) - matched,
        "spurious": len(predicted) - matched,
        "matched_iou_sum": float(sum(matched_ious)),
        "matched_count": len(matched_ious),
    }


def safe_f1(precision, recall):
    if pd.isna(precision) or pd.isna(recall):
        return np.nan
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_entity_metrics(
    items: pd.DataFrame,
    reference_method: str,
    output_dir: Path,
):
    items = items.copy()
    items["global_row_id"] = items["global_row_id"].astype(str)
    items["method_id"] = items["method_id"].astype(str)

    if reference_method not in set(items["method_id"]):
        raise ValueError(
            f"Reference method {reference_method!r} is not present in study_items.csv."
        )

    lookup = {
        (row.global_row_id, row.method_id): parse_spans_json(row.spans_json)
        for row in items.itertuples(index=False)
    }

    argument_rows = []
    for global_row_id in sorted(items["global_row_id"].unique()):
        reference = lookup[(global_row_id, reference_method)]

        for method in METHOD_ORDER:
            if (global_row_id, method) not in lookup:
                continue
            predicted = lookup[(global_row_id, method)]
            counts = ner_style_span_counts(predicted, reference)
            argument_rows.append(
                {
                    "global_row_id": global_row_id,
                    "method_id": method,
                    "reference_method": reference_method,
                    **counts,
                }
            )

    by_argument = pd.DataFrame(argument_rows)
    by_argument.to_csv(
        output_dir / f"entity_span_metrics_by_argument_against_{reference_method}.csv",
        index=False,
    )

    summary_rows = []
    for method, group in by_argument.groupby("method_id"):
        exact = group["exact"].sum()
        partial = group["partial"].sum()
        predicted_count = group["predicted"].sum()
        reference_count = group["reference"].sum()

        exact_precision = exact / predicted_count if predicted_count else np.nan
        exact_recall = exact / reference_count if reference_count else np.nan

        # SemEval-style partial boundary evaluation assigns half credit to a
        # one-to-one partial overlap after exact matches have been removed.
        partial_credit = exact + 0.5 * partial
        partial_precision = (
            partial_credit / predicted_count if predicted_count else np.nan
        )
        partial_recall = (
            partial_credit / reference_count if reference_count else np.nan
        )

        matched_count = group["matched_count"].sum()
        mean_matched_iou = (
            group["matched_iou_sum"].sum() / matched_count
            if matched_count
            else np.nan
        )

        summary_rows.append(
            {
                "method_id": method,
                "reference_method": reference_method,
                "n_arguments": group["global_row_id"].nunique(),
                "predicted_entities": predicted_count,
                "reference_entities": reference_count,
                "exact_matches": exact,
                "partial_matches": partial,
                "exact_precision": exact_precision,
                "exact_recall": exact_recall,
                "exact_f1": safe_f1(exact_precision, exact_recall),
                "partial_precision": partial_precision,
                "partial_recall": partial_recall,
                "partial_f1": safe_f1(partial_precision, partial_recall),
                "mean_iou_of_matched_spans": mean_matched_iou,
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        output_dir / f"entity_span_metrics_against_{reference_method}.csv",
        index=False,
    )


def create_plots(
    participant_method: pd.DataFrame,
    method_summary: pd.DataFrame,
    pairwise_matrix: pd.DataFrame,
    output_dir: Path,
):
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    # Completeness, precision, and the exploratory harmonic composite.
    plot_metrics = [
        ("completeness", "Completeness", (1, 7)),
        ("precision", "Precision", (1, 7)),
        ("human_f1_like_1_7", "F1-like balanced quality", (1, 7)),
    ]

    grouped = []
    for method in METHOD_ORDER:
        method_df = participant_method[
            participant_method["method_id"] == method
        ]
        row = {"method": METHOD_LABELS.get(method, method)}
        for metric, _, _ in plot_metrics:
            row[metric] = method_df[metric].mean()
        grouped.append(row)
    grouped = pd.DataFrame(grouped)

    x = np.arange(len(grouped))
    width = 0.25
    plt.figure(figsize=(13, 6))
    for i, (metric, label, _) in enumerate(plot_metrics):
        offset = (i - 1) * width
        plt.bar(x + offset, grouped[metric], width=width, label=label)
    plt.xticks(x, grouped["method"], rotation=20, ha="right")
    plt.ylim(1, 7)
    plt.ylabel("Mean participant-level score")
    plt.xlabel("Method")
    plt.title("Human span evaluation by method")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        figure_dir / "human_scores_by_method.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    for metric, label, ylim in plot_metrics:
        data = []
        labels = []
        for method in METHOD_ORDER:
            values = participant_method.loc[
                participant_method["method_id"] == method,
                metric,
            ].dropna()
            if len(values):
                data.append(values.to_numpy())
                labels.append(METHOD_LABELS.get(method, method))

        plt.figure(figsize=(11, 6))
        plt.boxplot(data, tick_labels=labels, showmeans=True)
        plt.ylim(*ylim)
        plt.ylabel(label)
        plt.xlabel("Method")
        plt.title(f"Participant-level {label} by method")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(
            figure_dir / f"{metric}_boxplot.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    ranking_summary = (
        participant_method.groupby("method_id", as_index=False)
        .agg(
            mean_rank=("rank", "mean"),
            first_place_rate=("is_first_rank", "mean"),
        )
        .set_index("method_id")
        .reindex(METHOD_ORDER)
        .reset_index()
    )
    ranking_summary["method"] = ranking_summary["method_id"].map(METHOD_LABELS)

    plt.figure(figsize=(10, 5.5))
    plt.bar(ranking_summary["method"], ranking_summary["mean_rank"])
    plt.ylabel("Mean rank (lower is better)")
    plt.xlabel("Method")
    plt.title("Mean ranking position by method")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(
        figure_dir / "mean_rank_by_method.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    plt.figure(figsize=(10, 5.5))
    plt.bar(ranking_summary["method"], ranking_summary["first_place_rate"])
    plt.ylabel("First-place rate")
    plt.xlabel("Method")
    plt.title("Share of displayed explanations ranked first")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(
        figure_dir / "first_place_rate_by_method.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    plt.figure(figsize=(8, 7))
    image = plt.imshow(
        pairwise_matrix.reindex(index=METHOD_ORDER, columns=METHOD_ORDER).to_numpy(dtype=float),
        vmin=0,
        vmax=1,
        aspect="auto",
    )
    plt.colorbar(image, label="Row method win rate")
    plt.xticks(
        np.arange(len(METHOD_ORDER)),
        [METHOD_LABELS[m] for m in METHOD_ORDER],
        rotation=35,
        ha="right",
    )
    plt.yticks(
        np.arange(len(METHOD_ORDER)),
        [METHOD_LABELS[m] for m in METHOD_ORDER],
    )
    plt.xlabel("Compared with")
    plt.ylabel("Method")
    plt.title("Pairwise ranking win rates")
    plt.tight_layout()
    plt.savefig(
        figure_dir / "pairwise_ranking_win_rates.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    responses = pd.read_csv(args.responses)
    mapping = pd.read_csv(args.mapping)
    arguments = (
        pd.read_csv(args.arguments)
        if args.arguments.exists()
        else None
    )

    ratings, rankings = reshape_responses(
        responses,
        mapping,
        arguments,
    )

    ratings_path = args.output_dir / "human_ratings_long.csv"
    rankings_path = args.output_dir / "argument_rankings_long.csv"
    ratings.to_csv(ratings_path, index=False)
    rankings.to_csv(rankings_path, index=False)

    agreement = compute_inter_rater_agreement(
        ratings,
        args.output_dir,
    )
    agreement.to_csv(
        args.output_dir / "inter_rater_agreement.csv",
        index=False,
    )

    argument_comments = (
        ratings[
            [
                "response_id",
                "version",
                "display_position",
                "argument_index",
                "global_row_id",
                "issue",
                "post_text",
                "comment",
            ]
        ]
        .drop_duplicates(
            ["response_id", "global_row_id"]
        )
        .copy()
    )

    argument_comments = argument_comments[
        argument_comments["comment"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
    ]

    argument_comments.to_csv(
        args.output_dir / "argument_comments.csv",
        index=False,
    )

    response_lookup = build_column_lookup(responses.columns)
    final_comment_col = find_column(
        response_lookup,
        "FINALCOMMENT",
    )

    response_id_col = next(
        (
            col
            for col in responses.columns
            if normalized_code(col) in {"RESPONSEID", "ID"}
        ),
        None,
    )

    if final_comment_col is not None:
        final_feedback = pd.DataFrame(
            {
                "response_id": (
                    responses[response_id_col]
                    if response_id_col is not None
                    else np.arange(1, len(responses) + 1)
                ),
                "final_feedback": responses[final_comment_col],
            }
        )

        final_feedback = final_feedback[
            final_feedback["final_feedback"]
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
        ]

        final_feedback.to_csv(
            args.output_dir / "final_feedback.csv",
            index=False,
        )

    participant_method, method_summary = build_method_summaries(
        ratings,
        args.output_dir,
    )
    _, pairwise_matrix = build_pairwise_ranking(
        ratings,
        args.output_dir,
    )
    exploratory_repeated_measures(
        participant_method,
        args.output_dir,
    )
    create_plots(
        participant_method,
        method_summary,
        pairwise_matrix,
        args.output_dir,
    )

    reference_method = args.entity_reference_method.strip().lower()
    if reference_method not in {"", "none", "off", "false"}:
        items = pd.read_csv(args.items)
        compute_entity_metrics(
            items,
            reference_method,
            args.output_dir,
        )

    print(f"Wrote {ratings_path}")
    print(f"Wrote {rankings_path}")
    print(f"Wrote {args.output_dir / 'method_summary.csv'}")
    print(f"Wrote plots to {args.output_dir / 'figures'}")
    print()
    print(
        f"Participants: {ratings['response_id'].nunique() if not ratings.empty else 0}"
    )
    print(f"Method ratings: {len(ratings)}")
    print(f"Ranked items: {len(rankings)}")
    print()
    print(
        "Note: human_f1_like is an exploratory harmonic composite of normalized "
        "Completeness and Precision ratings. It is F1-like, not a standard "
        "count-based precision/recall F1 score."
    )
    if reference_method not in {"", "none", "off", "false"}:
        print(
            f"Supplementary NER-style span metrics use {reference_method!r} "
            "as a reference and should not be interpreted as human gold labels."
        )


if __name__ == "__main__":
    main()
