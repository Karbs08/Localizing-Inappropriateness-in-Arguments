from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAPPING = REPO_ROOT / "human_study" / "survey_output" / "question_mapping.csv"
DEFAULT_ARGUMENTS = REPO_ROOT / "human_study" / "survey_output" / "selected_arguments.csv"
DEFAULT_OUTPUT = REPO_ROOT / "human_study" / "survey_results"

CRITERIA = ("REL", "SUF", "COM", "PRE")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reshape a LimeSurvey response export to tidy human-study tables."
    )
    parser.add_argument(
        "--responses",
        type=Path,
        required=True,
        help="CSV exported from LimeSurvey using question codes as headings.",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=DEFAULT_MAPPING,
        help="question_mapping.csv generated together with the survey.",
    )
    parser.add_argument(
        "--arguments",
        type=Path,
        default=DEFAULT_ARGUMENTS,
        help="selected_arguments.csv generated together with the survey.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    return parser.parse_args()


def normalized_code(value: str) -> str:
    """Normalize LimeSurvey qcode variants such as Q[REL], Q_REL, or Q.REL."""
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

    # Works for both exported answer codes ("4") and full labels ("4 – Partially").
    match = re.match(r"^\s*(\d+)", text)
    if match:
        return int(match.group(1))

    if text.upper() in {"NA", "N/A"} or text.lower().startswith("not assessable"):
        return pd.NA

    return pd.NA


def parse_preference(value):
    if pd.isna(value):
        return pd.NA

    text = str(value).strip()
    if not text:
        return pd.NA

    upper = text.upper()
    if upper in {"A", "B", "C", "N"}:
        return upper

    match = re.search(r"EXPLANATION\s+([ABC])", upper)
    if match:
        return match.group(1)

    if "NO MEANINGFUL" in upper or "NO PREFERENCE" in upper:
        return "N"

    return pd.NA


def main() -> None:
    args = parse_args()

    responses = pd.read_csv(args.responses)
    mapping = pd.read_csv(args.mapping)
    arguments = pd.read_csv(args.arguments) if args.arguments.exists() else None

    lookup = build_column_lookup(responses.columns)

    version_col = find_column(lookup, "VERSION")
    if version_col is None:
        raise KeyError(
            "Could not find VERSION. Export the LimeSurvey responses with "
            "'Question code' as the question heading (preferably with "
            "'Use ExpressionScript code' enabled)."
        )

    response_id_col = next(
        (col for col in responses.columns if normalized_code(col) in {"RESPONSEID", "ID"}),
        None,
    )
    seed_col = next(
        (col for col in responses.columns if normalized_code(col) == "SEED"),
        None,
    )

    mapping["version"] = mapping["version"].astype(int)
    mapping["global_row_id"] = mapping["global_row_id"].astype(str)

    explanation_rows: list[dict] = []

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

        # Determine the preferred true method once per argument.
        preference_by_argument: dict[tuple, tuple] = {}

        for _, group in participant_mapping.groupby(
            ["global_row_id", "display_position", "argument_index"],
            sort=False,
        ):
            first = group.iloc[0]
            pref_code = str(first["preference_question_code"])
            pref_col = find_column(lookup, pref_code)
            pref_label = (
                parse_preference(response[pref_col])
                if pref_col is not None
                else pd.NA
            )

            preferred_method = pd.NA
            if pd.notna(pref_label) and pref_label in {"A", "B", "C"}:
                match = group[group["display_label"] == pref_label]
                if not match.empty:
                    preferred_method = str(match.iloc[0]["method_id"])

            key = (
                str(first["global_row_id"]),
                int(first["display_position"]),
                int(first["argument_index"]),
            )
            preference_by_argument[key] = (pref_label, preferred_method)

        for _, item in participant_mapping.iterrows():
            rating_code = str(item["rating_question_code"])
            overall_code = str(item["overall_question_code"])

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

            overall_col = find_column(lookup, overall_code)
            overall = (
                parse_rating(response[overall_col])
                if overall_col is not None
                else pd.NA
            )

            key = (
                str(item["global_row_id"]),
                int(item["display_position"]),
                int(item["argument_index"]),
            )
            preferred_label, preferred_method = preference_by_argument[key]

            pref_code = str(item["preference_question_code"])
            comment_code = re.sub(r"P$", "CMT", pref_code)
            comment_col = find_column(lookup, comment_code)
            comment = (
                response[comment_col]
                if comment_col is not None and pd.notna(response[comment_col])
                else ""
            )

            explanation_rows.append(
                {
                    "response_id": response_id,
                    "version": version,
                    "lime_seed": lime_seed,
                    "display_position": int(item["display_position"]),
                    "argument_index": int(item["argument_index"]),
                    "global_row_id": str(item["global_row_id"]),
                    "method_id": str(item["method_id"]),
                    "display_label": str(item["display_label"]),
                    "overall_inappropriateness": overall,
                    "relevance": criterion_values["REL"],
                    "sufficiency": criterion_values["SUF"],
                    "completeness": criterion_values["COM"],
                    "precision": criterion_values["PRE"],
                    "preferred_display_label": preferred_label,
                    "preferred_method": preferred_method,
                    "is_preferred": (
                        pd.NA
                        if pd.isna(preferred_label) or preferred_label == "N"
                        else str(item["method_id"]) == preferred_method
                    ),
                    "comment": comment,
                }
            )

    ratings = pd.DataFrame(explanation_rows)

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

    args.output_dir.mkdir(parents=True, exist_ok=True)

    ratings_path = args.output_dir / "human_ratings_long.csv"
    ratings.to_csv(ratings_path, index=False)

    argument_cols = [
        "response_id",
        "version",
        "lime_seed",
        "display_position",
        "argument_index",
        "global_row_id",
        "overall_inappropriateness",
        "preferred_display_label",
        "preferred_method",
        "comment",
    ]
    for optional in ("issue", "post_text"):
        if optional in ratings.columns:
            argument_cols.append(optional)

    argument_ratings = (
        ratings[argument_cols]
        .drop_duplicates(
            ["response_id", "global_row_id", "display_position"]
        )
        .sort_values(["response_id", "display_position"])
    )

    arguments_path = args.output_dir / "argument_ratings.csv"
    argument_ratings.to_csv(arguments_path, index=False)

    print(f"Wrote {ratings_path}")
    print(f"Wrote {arguments_path}")
    print()
    print(
        f"{ratings['response_id'].nunique() if not ratings.empty else 0} responses, "
        f"{len(argument_ratings)} argument ratings, "
        f"{len(ratings)} explanation ratings."
    )


if __name__ == "__main__":
    main()
