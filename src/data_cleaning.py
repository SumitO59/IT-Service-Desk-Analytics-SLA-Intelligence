from pathlib import Path

import numpy as np
import pandas as pd


# -------------------------------------------------------------------
# Project paths
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_FILE = PROJECT_ROOT / "data" / "raw" / "incident_event_log.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = PROCESSED_DIR / "incidents_clean.csv"


# -------------------------------------------------------------------
# Dataset configuration
# -------------------------------------------------------------------

TIMESTAMP_COLUMNS = [
    "opened_at",
    "sys_updated_at",
    "resolved_at",
    "closed_at",
]

CUMULATIVE_COLUMNS = [
    "reassignment_count",
    "reopen_count",
    "sys_mod_count",
]


# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------

def normalize_boolean(series: pd.Series) -> pd.Series:
    """
    Convert common boolean representations into pandas nullable booleans.
    """

    normalized = (
        series
        .astype("string")
        .str.strip()
        .str.lower()
    )

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }

    result = normalized.map(mapping)

    unexpected = normalized[
        result.isna() & normalized.notna()
    ].unique()

    if len(unexpected) > 0:
        raise ValueError(
            f"Unexpected boolean values found in "
            f"'{series.name}': {unexpected.tolist()}"
        )

    return result.astype("boolean")


def create_reassignment_bucket(series: pd.Series) -> pd.Series:
    """
    Convert reassignment counts into analytical buckets.

    0     -> no reassignment
    1-2   -> low reassignment
    3-5   -> moderate reassignment
    6+    -> high reassignment
    """

    return pd.cut(
        series,
        bins=[-np.inf, 0, 2, 5, np.inf],
        labels=["0", "1-2", "3-5", "6+"],
        include_lowest=True,
    )


# -------------------------------------------------------------------
# Main transformation
# -------------------------------------------------------------------

def build_incident_level_dataset() -> pd.DataFrame:

    print("=" * 70)
    print("IT SERVICE DESK — INCIDENT-LEVEL DATA TRANSFORMATION")
    print("=" * 70)

    # ---------------------------------------------------------------
    # 1. Load raw event log
    # ---------------------------------------------------------------

    print("\n[1/7] Loading raw dataset...")

    df = pd.read_csv(
        RAW_FILE,
        na_values=["?"],
        keep_default_na=True,
        low_memory=False,
    )

    print(f"Raw rows: {len(df):,}")
    print(f"Raw columns: {len(df.columns)}")
    print(f"Unique incidents: {df['number'].nunique():,}")

    # ---------------------------------------------------------------
    # 2. Parse timestamp columns
    # ---------------------------------------------------------------

    print("\n[2/7] Parsing timestamp columns...")

    for column in TIMESTAMP_COLUMNS:
        df[column] = pd.to_datetime(
            df[column],
            errors="coerce",
            dayfirst=True,
        )

    print("Timestamp parsing complete.")

    # ---------------------------------------------------------------
    # 3. Normalize important fields
    # ---------------------------------------------------------------

    print("\n[3/7] Normalizing fields...")

    if "made_sla" in df.columns:
        df["made_sla"] = normalize_boolean(
            df["made_sla"]
        )

    if "active" in df.columns:
        df["active"] = normalize_boolean(
            df["active"]
        )

    for column in CUMULATIVE_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # ---------------------------------------------------------------
    # 4. Sort incident lifecycle events
    # ---------------------------------------------------------------

    print("\n[4/7] Sorting incident lifecycle events...")

    # Preserve original row order as a deterministic tie-breaker
    # when two events have the same sys_updated_at timestamp.
    df["_original_row_order"] = np.arange(len(df))

    df = df.sort_values(
        by=[
            "number",
            "sys_updated_at",
            "_original_row_order",
        ],
        ascending=[
            True,
            True,
            True,
        ],
        kind="mergesort",
    )

    # ---------------------------------------------------------------
    # 5. Aggregate event records to incident level
    # ---------------------------------------------------------------

    print(
        "\n[5/7] Aggregating event records "
        "to incident level..."
    )

    grouped = df.groupby(
        "number",
        sort=False,
    )

    # The final event represents the latest lifecycle state.
    latest_events = grouped.tail(1).copy()

    # opened_at is consistent within each incident according
    # to our previous audit.
    opened_at = (
        grouped["opened_at"]
        .first()
        .rename("opened_at")
    )

    # Cumulative counters use their maximum lifecycle value.
    cumulative_max = grouped[
        CUMULATIVE_COLUMNS
    ].max()

    incidents = latest_events.set_index(
        "number"
    )

    # Explicitly use the incident's original opened timestamp.
    incidents["opened_at"] = opened_at

    # Replace cumulative counters with their maximum values.
    for column in CUMULATIVE_COLUMNS:
        incidents[column] = cumulative_max[column]

    # ---------------------------------------------------------------
    # 6. Calculate analytical metrics
    # ---------------------------------------------------------------

    print(
        "\n[6/7] Calculating analytical metrics..."
    )

    # Final made_sla=False means the incident breached SLA.
    incidents["sla_breached"] = (
        incidents["made_sla"].eq(False)
    )

    # An incident is considered reopened if reopen_count > 0.
    incidents["reopened"] = (
        incidents["reopen_count"]
        .fillna(0)
        .gt(0)
    )

    # Reassignment bucket.
    incidents["reassignment_bucket"] = (
        create_reassignment_bucket(
            incidents["reassignment_count"]
        )
    )

    # ---------------------------------------------------------------
    # Resolution time
    # ---------------------------------------------------------------

    incidents["resolution_time_hours"] = (
        incidents["resolved_at"]
        - incidents["opened_at"]
    ).dt.total_seconds() / 3600

    # ---------------------------------------------------------------
    # Closure time
    # ---------------------------------------------------------------

    incidents["closure_time_hours"] = (
        incidents["closed_at"]
        - incidents["opened_at"]
    ).dt.total_seconds() / 3600

    # ---------------------------------------------------------------
    # Calendar dimensions
    # ---------------------------------------------------------------

    incidents["opened_date"] = (
        incidents["opened_at"].dt.date
    )

    incidents["opened_month"] = (
        incidents["opened_at"]
        .dt.to_period("M")
        .astype("string")
    )

    incidents["opened_year"] = (
        incidents["opened_at"].dt.year
    )

    # ---------------------------------------------------------------
    # 7. Finalize and export
    # ---------------------------------------------------------------

    print(
        "\n[7/7] Finalizing and exporting dataset..."
    )

    incidents = incidents.reset_index()

    if "_original_row_order" in incidents.columns:
        incidents = incidents.drop(
            columns=["_original_row_order"]
        )

    # Put important analytical columns first.
    preferred_columns = [
        "number",
        "opened_at",
        "resolved_at",
        "closed_at",
        "resolution_time_hours",
        "closure_time_hours",
        "made_sla",
        "sla_breached",
        "reassignment_count",
        "reassignment_bucket",
        "reopen_count",
        "reopened",
        "sys_mod_count",
        "incident_state",
        "active",
        "category",
        "subcategory",
        "priority",
        "impact",
        "urgency",
        "assignment_group",
        "assigned_to",
        "caller_id",
        "contact_type",
        "location",
        "opened_date",
        "opened_month",
        "opened_year",
    ]

    remaining_columns = [
        column
        for column in incidents.columns
        if column not in preferred_columns
    ]

    incidents = incidents[
        [
            column
            for column in preferred_columns
            if column in incidents.columns
        ]
        + remaining_columns
    ]

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    incidents.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\nTransformation complete.")
    print(f"Output file: {OUTPUT_FILE}")
    print(
        f"Incident-level rows: "
        f"{len(incidents):,}"
    )
    print(
        f"Columns: "
        f"{len(incidents.columns)}"
    )

    return incidents


# -------------------------------------------------------------------
# Validation
# -------------------------------------------------------------------

def validate_incident_dataset(
    incidents: pd.DataFrame,
) -> None:

    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Structural validation
    # ---------------------------------------------------------------

    row_count = len(incidents)

    unique_ids = incidents[
        "number"
    ].nunique()

    duplicate_ids = incidents[
        "number"
    ].duplicated().sum()

    print(f"\nRows: {row_count:,}")
    print(
        f"Unique incident IDs: "
        f"{unique_ids:,}"
    )
    print(
        f"Duplicate incident IDs: "
        f"{duplicate_ids:,}"
    )

    expected_rows = 24_918

    if row_count != expected_rows:
        raise AssertionError(
            f"Expected {expected_rows:,} incident rows, "
            f"but found {row_count:,}."
        )

    if unique_ids != expected_rows:
        raise AssertionError(
            f"Expected {expected_rows:,} unique incidents, "
            f"but found {unique_ids:,}."
        )

    if duplicate_ids != 0:
        raise AssertionError(
            f"Found {duplicate_ids:,} duplicate incident IDs."
        )

    # ---------------------------------------------------------------
    # opened_at validation
    # ---------------------------------------------------------------

    missing_opened = (
        incidents["opened_at"]
        .isna()
        .sum()
    )

    print(
        f"Missing opened_at: "
        f"{missing_opened:,}"
    )

    if missing_opened != 0:
        raise AssertionError(
            "opened_at should be present "
            "for every incident."
        )

    # ---------------------------------------------------------------
    # Resolution and closure validation
    # ---------------------------------------------------------------

    missing_resolved = (
        incidents["resolved_at"]
        .isna()
        .sum()
    )

    negative_resolution = incidents[
        incidents["resolution_time_hours"] < 0
    ]

    negative_closure = incidents[
        incidents["closure_time_hours"] < 0
    ]

    print(
        f"\nMissing resolved_at: "
        f"{missing_resolved:,}"
    )

    print(
        f"Negative resolution times: "
        f"{len(negative_resolution):,}"
    )

    print(
        f"Negative closure times: "
        f"{len(negative_closure):,}"
    )

    if len(negative_resolution) > 0:

        print(
            "\nWARNING: Negative "
            "resolution times detected."
        )

        print(
            negative_resolution[
                [
                    "number",
                    "opened_at",
                    "resolved_at",
                    "resolution_time_hours",
                ]
            ]
            .head(10)
            .to_string(index=False)
        )

    if len(negative_closure) > 0:

        print(
            "\nWARNING: Negative "
            "closure times detected."
        )

        print(
            negative_closure[
                [
                    "number",
                    "opened_at",
                    "closed_at",
                    "closure_time_hours",
                ]
            ]
            .head(10)
            .to_string(index=False)
        )

    # ---------------------------------------------------------------
    # SLA validation
    # ---------------------------------------------------------------

    print("\nSLA distribution:")

    print(
        incidents["made_sla"]
        .value_counts(
            dropna=False
        )
        .sort_index()
    )

    print("\nSLA breach distribution:")

    print(
        incidents["sla_breached"]
        .value_counts(
            dropna=False
        )
        .sort_index()
    )

    # ---------------------------------------------------------------
    # Reassignment validation
    # ---------------------------------------------------------------

    print("\nReassignment buckets:")

    print(
        incidents["reassignment_bucket"]
        .value_counts(
            dropna=False
        )
        .sort_index()
    )

    # ---------------------------------------------------------------
    # Reopen validation
    # ---------------------------------------------------------------

    print("\nReopened incidents:")

    print(
        incidents["reopened"]
        .value_counts(
            dropna=False
        )
        .sort_index()
    )

    print("\nValidation complete.")


# -------------------------------------------------------------------
# Script entry point
# -------------------------------------------------------------------

if __name__ == "__main__":

    incidents_df = (
        build_incident_level_dataset()
    )

    validate_incident_dataset(
        incidents_df
    )
