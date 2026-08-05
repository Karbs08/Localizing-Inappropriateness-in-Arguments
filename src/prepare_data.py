"""Prepare the Appropriateness Corpus for all span-localization experiments.

This script downloads the original Hugging Face dataset once, stores a local
raw copy, creates stable row identifiers, applies the shared text
normalization, computes the original classifier predictions, and saves the
result as a reusable Parquet file.

Run this script from the project root:

    python -m utils.prepare_data

The generated processed dataset should be treated as the canonical input for
all experiment notebooks. It must be regenerated whenever the dataset,
classifier model, preprocessing logic, or relevant inference settings change.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, load_from_disk
from transformers import pipeline

from src.utils import (
    add_classifier_outputs,
    attach_split,
    normalize_text,
)


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "appropriateness_corpus"
)

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DATA_FILE = (
    PROCESSED_DATA_DIR
    / "appropriateness_prepared.parquet"
)

METADATA_FILE = (
    PROCESSED_DATA_DIR
    / "appropriateness_prepared_metadata.json"
)


# ---------------------------------------------------------------------
# Dataset and model configuration
# ---------------------------------------------------------------------

DATASET_NAME = "timonziegenbein/appropriateness-corpus"
MODEL_NAME = "timonziegenbein/appropriateness-classifier-binary"

TEXT_COL = "post_text"
TEXT_NORM_COL = "text_norm"
LABEL_COL = "Inappropriateness"

INAPPROPRIATE_LABEL = "LABEL_1"

MAX_LENGTH = 512
BATCH_SIZE = 32

if torch.cuda.is_available():
    DEVICE = 0          # Cuda
elif torch.backends.mps.is_available():
    DEVICE = "mps"      # Apple-Silicon-GPU
else:
    DEVICE = -1         # CPU


def load_or_download_raw_dataset():
    """Load the local raw dataset or download and store it once."""
    if RAW_DATA_DIR.exists():
        print(f"Loading local raw dataset from: {RAW_DATA_DIR}")

        return load_from_disk(str(RAW_DATA_DIR))

    print(f"Downloading dataset: {DATASET_NAME}")

    dataset = load_dataset(DATASET_NAME)

    RAW_DATA_DIR.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset.save_to_disk(str(RAW_DATA_DIR))

    print(f"Saved raw dataset to: {RAW_DATA_DIR}")

    return dataset


def build_canonical_dataframe(dataset) -> pd.DataFrame:
    """Combine all original splits and assign stable global row IDs."""
    required_splits = [
        "train",
        "validation",
        "test",
    ]

    missing_splits = [
        split
        for split in required_splits
        if split not in dataset
    ]

    if missing_splits:
        raise KeyError(
            "Dataset is missing required splits: "
            + ", ".join(missing_splits)
        )

    split_frames = []

    for split_name in required_splits:
        split_df = dataset[split_name].to_pandas()

        split_df = attach_split(
            split_df,
            split_name,
        )

        split_frames.append(split_df)

    df_all = pd.concat(
        split_frames,
        ignore_index=True,
    )

    # Assign the ID only after concatenating the splits. This guarantees
    # identical IDs across all experiment methods.
    df_all["global_row_id"] = np.arange(
        len(df_all),
        dtype=np.int64,
    )

    df_all[TEXT_NORM_COL] = (
        df_all[TEXT_COL]
        .apply(normalize_text)
    )

    return df_all


def create_classifier():
    """Initialize the document-level appropriateness classifier."""
    return pipeline(
        "text-classification",
        model=MODEL_NAME,
        device=DEVICE,
        top_k=None,
        truncation=True,
        max_length=MAX_LENGTH,
    )


def save_metadata(
    dataset,
    df_all: pd.DataFrame,
) -> None:
    """Save preprocessing information required for reproducibility."""
    metadata = {
        "dataset_name": DATASET_NAME,
        "model_name": MODEL_NAME,
        "text_column": TEXT_COL,
        "normalized_text_column": TEXT_NORM_COL,
        "label_column": LABEL_COL,
        "inappropriate_label": INAPPROPRIATE_LABEL,
        "max_length": MAX_LENGTH,
        "number_of_rows": len(df_all),
        "split_sizes": (
            df_all["split"]
            .value_counts()
            .sort_index()
            .to_dict()
        ),
        "dataset_fingerprints": {
            split_name: dataset[split_name]._fingerprint
            for split_name in dataset
        },
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    with METADATA_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )


def main() -> None:
    """Create and save the canonical experiment dataset."""
    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset = load_or_download_raw_dataset()

    print("Building canonical dataframe...")
    df_all = build_canonical_dataframe(dataset)

    print("Loading classifier...")
    classifier = create_classifier()

    print("Computing original classifier predictions...")
    df_all = add_classifier_outputs(
        df_all,
        classifier=classifier,
        text_col=TEXT_NORM_COL,
        label_col=LABEL_COL,
        inappropriate_label=INAPPROPRIATE_LABEL,
        batch_size=BATCH_SIZE,
    )

    print(f"Saving processed dataset to: {PROCESSED_DATA_FILE}")

    df_all.to_parquet(
        PROCESSED_DATA_FILE,
        index=False,
    )

    save_metadata(
        dataset,
        df_all,
    )

    print()
    print("Prepared dataset successfully.")
    print(f"Rows: {len(df_all)}")
    print()
    print("Split sizes:")
    print(df_all["split"].value_counts().sort_index())
    print()
    print("Confusion types:")
    print(df_all["confusion_type"].value_counts().sort_index())


if __name__ == "__main__":
    main()