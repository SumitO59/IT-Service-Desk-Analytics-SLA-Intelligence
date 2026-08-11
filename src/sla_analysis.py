"""
SLA Risk Analysis

Milestone 3:
- Category SLA performance
- Priority SLA performance
- Reassignment relationships
- Assignment-group performance
- Operational bottleneck identification
- SLA-risk segmentation

This module builds on the incident-level dataset created during
Milestone 1 and the analytical metrics established during Milestone 2.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "incidents_clean.csv"

REPORT_DIR = PROJECT_ROOT / "reports"

CATEGORY_OUTPUT = REPORT_DIR / "category_sla_performance.csv"
PRIORITY_OUTPUT = REPORT_DIR / "priority_sla_performance.csv"
REASSIGNMENT_OUTPUT = REPORT_DIR / "reassignment_sla_performance.csv"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MIN_CATEGORY_SAMPLE = 100
CONFIDENCE_Z = 1.96


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def wilson_interval(
    successes: int,
    total: int,
    z: float = CONFIDENCE_Z,
) -> tuple[float, float]:
    """
    Calculate a Wilson score confidence interval for a binomial proportion.

    Parameters
    ----------
    successes:
        Number of SLA-breached incidents.

    total:
        Total incidents in the group.

    z:
        Normal critical value for the desired confidence level.
        1.96 corresponds approximately to a 95% confidence interval.

    Returns
    -------
    tuple[float, float]
        Lower and upper bounds of the confidence interval.
    """

    if total == 0:
        return np.nan, np.nan

    proportion = successes / total

    denominator = 1 + (z**2 / total)

    center = (
        proportion
        + (z**2 / (2 * total))
    ) / denominator

    margin = (
        z
        * np.sqrt(
            (
                proportion * (1 - proportion) / total
            )
            + (z**2 / (4 * total**2))
        )
        / denominator
    )

    lower = center - margin
    upper = center + margin

    # A binomial proportion cannot fall outside [0, 1].
    # Clipping prevents floating-point errors from producing
    # impossible bounds such as -2.77e-17.
    lower = max(0.0, lower)
    upper = min(1.0, upper)

    return lower, upper


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_incident_data() -> pd.DataFrame:
    """
    Load the cleaned incident-level dataset.
    """

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False,
    )

    required_columns = [
        "number",
        "category",
        "priority",
        "made_sla",
        "sla_breached",
        "resolution_time_hours",
        "reassignment_count",
        "reassignment_bucket",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Required columns missing from dataset: "
            + ", ".join(missing_columns)
        )

    return df


# ---------------------------------------------------------------------------
# Category SLA analysis
# ---------------------------------------------------------------------------

def analyze_category_sla(
    df: pd.DataFrame,
    min_sample: int = MIN_CATEGORY_SAMPLE,
) -> pd.DataFrame:
    """
    Calculate SLA-risk metrics by incident category.

    Categories with fewer than `min_sample` incidents are retained in the
    output but marked as ineligible for primary comparison.
    """

    category_df = df.copy()

    category_df = category_df.dropna(
        subset=["category"]
    )

    grouped = (
        category_df
        .groupby("category", observed=True)
        .agg(
            incident_count=("number", "nunique"),
            sla_breaches=("sla_breached", "sum"),
            median_resolution_hours=(
                "resolution_time_hours",
                "median",
            ),
            p90_resolution_hours=(
                "resolution_time_hours",
                lambda x: x.quantile(0.90),
            ),
            reassigned_incidents=(
                "reassignment_count",
                lambda x: (x > 0).sum(),
            ),
            mean_reassignment_count=(
                "reassignment_count",
                "mean",
            ),
        )
        .reset_index()
    )

    grouped["sla_breaches"] = (
        grouped["sla_breaches"].astype(int)
    )

    grouped["breach_rate"] = (
        grouped["sla_breaches"]
        / grouped["incident_count"]
    )

    grouped["sla_compliance"] = (
        1 - grouped["breach_rate"]
    )

    grouped["reassignment_rate"] = (
        grouped["reassigned_incidents"]
        / grouped["incident_count"]
    )

    intervals = grouped.apply(
        lambda row: wilson_interval(
            successes=int(row["sla_breaches"]),
            total=int(row["incident_count"]),
        ),
        axis=1,
    )

    grouped["breach_rate_ci_lower"] = intervals.apply(
        lambda interval: interval[0]
    )

    grouped["breach_rate_ci_upper"] = intervals.apply(
        lambda interval: interval[1]
    )

    grouped["eligible_for_comparison"] = (
        grouped["incident_count"] >= min_sample
    )

    percentage_columns = [
        "breach_rate",
        "sla_compliance",
        "reassignment_rate",
        "breach_rate_ci_lower",
        "breach_rate_ci_upper",
    ]

    for column in percentage_columns:
        grouped[column] = grouped[column] * 100

    grouped = grouped.sort_values(
        by=[
            "eligible_for_comparison",
            "breach_rate",
            "incident_count",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    ).reset_index(
        drop=True
    )

    return grouped


# ---------------------------------------------------------------------------
# Priority SLA analysis
# ---------------------------------------------------------------------------

def analyze_priority_sla(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate SLA-risk metrics by incident priority.

    Priority is treated as an explanatory segmentation variable.
    Results describe association with SLA outcomes and do not imply causality.
    """

    priority_df = df.copy()

    priority_df = priority_df.dropna(
        subset=["priority"]
    )

    grouped = (
        priority_df
        .groupby("priority", observed=True)
        .agg(
            incident_count=("number", "nunique"),
            sla_breaches=("sla_breached", "sum"),
            median_resolution_hours=(
                "resolution_time_hours",
                "median",
            ),
            p90_resolution_hours=(
                "resolution_time_hours",
                lambda x: x.quantile(0.90),
            ),
            reassigned_incidents=(
                "reassignment_count",
                lambda x: (x > 0).sum(),
            ),
            mean_reassignment_count=(
                "reassignment_count",
                "mean",
            ),
        )
        .reset_index()
    )

    grouped["sla_breaches"] = (
        grouped["sla_breaches"].astype(int)
    )

    grouped["breach_rate"] = (
        grouped["sla_breaches"]
        / grouped["incident_count"]
    )

    grouped["sla_compliance"] = (
        1 - grouped["breach_rate"]
    )

    grouped["reassignment_rate"] = (
        grouped["reassigned_incidents"]
        / grouped["incident_count"]
    )

    intervals = grouped.apply(
        lambda row: wilson_interval(
            successes=int(row["sla_breaches"]),
            total=int(row["incident_count"]),
        ),
        axis=1,
    )

    grouped["breach_rate_ci_lower"] = intervals.apply(
        lambda interval: interval[0]
    )

    grouped["breach_rate_ci_upper"] = intervals.apply(
        lambda interval: interval[1]
    )

    percentage_columns = [
        "breach_rate",
        "sla_compliance",
        "reassignment_rate",
        "breach_rate_ci_lower",
        "breach_rate_ci_upper",
    ]

    for column in percentage_columns:
        grouped[column] = grouped[column] * 100

    priority_order = [
        "1 - Critical",
        "2 - High",
        "3 - Moderate",
        "4 - Low",
    ]

    grouped["priority_order"] = (
        grouped["priority"].map(
            {
                value: index
                for index, value in enumerate(priority_order)
            }
        )
    )

    grouped = (
        grouped
        .sort_values("priority_order")
        .drop(columns="priority_order")
        .reset_index(drop=True)
    )

    return grouped


# ---------------------------------------------------------------------------
# Reassignment vs SLA analysis
# ---------------------------------------------------------------------------

def analyze_reassignment_sla(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate SLA performance across reassignment buckets.

    The existing reassignment buckets created during data cleaning are used
    directly rather than reconstructing them independently.
    """

    reassignment_df = df.copy()

    reassignment_df = reassignment_df.dropna(
        subset=["reassignment_bucket"]
    )

    grouped = (
        reassignment_df
        .groupby(
            "reassignment_bucket",
            observed=True,
        )
        .agg(
            incident_count=("number", "nunique"),
            sla_breaches=("sla_breached", "sum"),
            median_resolution_hours=(
                "resolution_time_hours",
                "median",
            ),
            p90_resolution_hours=(
                "resolution_time_hours",
                lambda x: x.quantile(0.90),
            ),
            mean_reassignment_count=(
                "reassignment_count",
                "mean",
            ),
        )
        .reset_index()
    )

    grouped["sla_breaches"] = (
        grouped["sla_breaches"].astype(int)
    )

    grouped["breach_rate"] = (
        grouped["sla_breaches"]
        / grouped["incident_count"]
    )

    grouped["sla_compliance"] = (
        1 - grouped["breach_rate"]
    )

    intervals = grouped.apply(
        lambda row: wilson_interval(
            successes=int(row["sla_breaches"]),
            total=int(row["incident_count"]),
        ),
        axis=1,
    )

    grouped["breach_rate_ci_lower"] = intervals.apply(
        lambda interval: interval[0]
    )

    grouped["breach_rate_ci_upper"] = intervals.apply(
        lambda interval: interval[1]
    )

    percentage_columns = [
        "breach_rate",
        "sla_compliance",
        "breach_rate_ci_lower",
        "breach_rate_ci_upper",
    ]

    for column in percentage_columns:
        grouped[column] = grouped[column] * 100

    bucket_order = [
        "0",
        "1-2",
        "3-5",
        "6+",
    ]

    grouped["bucket_order"] = (
        grouped["reassignment_bucket"].map(
            {
                value: index
                for index, value in enumerate(bucket_order)
            }
        )
    )

    grouped = (
        grouped
        .sort_values("bucket_order")
        .drop(columns="bucket_order")
        .reset_index(drop=True)
    )

    return grouped


# ---------------------------------------------------------------------------
# Category validation
# ---------------------------------------------------------------------------

def validate_category_analysis(
    result: pd.DataFrame,
    source_df: pd.DataFrame,
) -> None:
    """
    Validate that category-level aggregation is internally consistent.
    """

    source_categories = (
        source_df["category"]
        .dropna()
        .nunique()
    )

    result_categories = result["category"].nunique()

    if source_categories != result_categories:
        raise AssertionError(
            "Category count mismatch: "
            f"source={source_categories}, "
            f"result={result_categories}"
        )

    total_category_incidents = (
        result["incident_count"].sum()
    )

    source_category_incidents = (
        source_df["category"].notna().sum()
    )

    if total_category_incidents != source_category_incidents:
        raise AssertionError(
            "Incident-count mismatch after category aggregation: "
            f"source={source_category_incidents}, "
            f"result={total_category_incidents}"
        )

    invalid_intervals = result[
        (
            result["breach_rate"]
            < result["breach_rate_ci_lower"]
        )
        |
        (
            result["breach_rate"]
            > result["breach_rate_ci_upper"]
        )
    ]

    if not invalid_intervals.empty:
        raise AssertionError(
            "Some breach rates fall outside "
            "their confidence intervals."
        )

    if (
        (result["breach_rate_ci_lower"] < 0).any()
        or
        (result["breach_rate_ci_upper"] > 100).any()
    ):
        raise AssertionError(
            "Invalid Wilson confidence interval bounds detected."
        )

    expected_eligibility = (
        result["incident_count"]
        >= MIN_CATEGORY_SAMPLE
    )

    if not (
        result["eligible_for_comparison"]
        == expected_eligibility
    ).all():
        raise AssertionError(
            "Sample-size eligibility flag is inconsistent."
        )


# ---------------------------------------------------------------------------
# Priority validation
# ---------------------------------------------------------------------------

def validate_priority_analysis(
    result: pd.DataFrame,
    source_df: pd.DataFrame,
) -> None:
    """
    Validate priority-level aggregation.
    """

    source_priorities = (
        source_df["priority"]
        .dropna()
        .nunique()
    )

    result_priorities = result["priority"].nunique()

    if source_priorities != result_priorities:
        raise AssertionError(
            "Priority count mismatch: "
            f"source={source_priorities}, "
            f"result={result_priorities}"
        )

    source_priority_incidents = (
        source_df["priority"].notna().sum()
    )

    result_priority_incidents = (
        result["incident_count"].sum()
    )

    if source_priority_incidents != result_priority_incidents:
        raise AssertionError(
            "Priority incident-count mismatch: "
            f"source={source_priority_incidents}, "
            f"result={result_priority_incidents}"
        )

    source_priority_breaches = (
        source_df.loc[
            source_df["priority"].notna(),
            "sla_breached",
        ].sum()
    )

    result_priority_breaches = (
        result["sla_breaches"].sum()
    )

    if source_priority_breaches != result_priority_breaches:
        raise AssertionError(
            "Priority SLA-breach mismatch: "
            f"source={source_priority_breaches}, "
            f"result={result_priority_breaches}"
        )

    invalid_intervals = result[
        (
            result["breach_rate"]
            < result["breach_rate_ci_lower"]
        )
        |
        (
            result["breach_rate"]
            > result["breach_rate_ci_upper"]
        )
    ]

    if not invalid_intervals.empty:
        raise AssertionError(
            "Some priority breach rates fall "
            "outside their confidence intervals."
        )

    if (
        (result["breach_rate_ci_lower"] < 0).any()
        or
        (result["breach_rate_ci_upper"] > 100).any()
    ):
        raise AssertionError(
            "Invalid priority Wilson confidence intervals."
        )


# ---------------------------------------------------------------------------
# Reassignment validation
# ---------------------------------------------------------------------------

def validate_reassignment_analysis(
    result: pd.DataFrame,
    source_df: pd.DataFrame,
) -> None:
    """
    Validate reassignment-bucket aggregation.
    """

    source_buckets = (
        source_df["reassignment_bucket"]
        .dropna()
        .nunique()
    )

    result_buckets = (
        result["reassignment_bucket"].nunique()
    )

    if source_buckets != result_buckets:
        raise AssertionError(
            "Reassignment bucket count mismatch: "
            f"source={source_buckets}, "
            f"result={result_buckets}"
        )

    source_incidents = (
        source_df["reassignment_bucket"]
        .notna()
        .sum()
    )

    result_incidents = (
        result["incident_count"].sum()
    )

    if source_incidents != result_incidents:
        raise AssertionError(
            "Reassignment incident-count mismatch: "
            f"source={source_incidents}, "
            f"result={result_incidents}"
        )

    source_breaches = (
        source_df.loc[
            source_df["reassignment_bucket"].notna(),
            "sla_breached",
        ].sum()
    )

    result_breaches = (
        result["sla_breaches"].sum()
    )

    if source_breaches != result_breaches:
        raise AssertionError(
            "Reassignment SLA-breach mismatch: "
            f"source={source_breaches}, "
            f"result={result_breaches}"
        )

    invalid_intervals = result[
        (
            result["breach_rate"]
            < result["breach_rate_ci_lower"]
        )
        |
        (
            result["breach_rate"]
            > result["breach_rate_ci_upper"]
        )
    ]

    if not invalid_intervals.empty:
        raise AssertionError(
            "Some reassignment breach rates fall "
            "outside their confidence intervals."
        )

    if (
        (result["breach_rate_ci_lower"] < 0).any()
        or
        (result["breach_rate_ci_upper"] > 100).any()
    ):
        raise AssertionError(
            "Invalid reassignment Wilson confidence intervals."
        )

    # Verify that the existing bucket definitions agree with the
    # underlying reassignment_count values.

    expected_bucket = pd.Series(
        np.select(
            [
                source_df["reassignment_count"] == 0,
                source_df["reassignment_count"].between(1, 2),
                source_df["reassignment_count"].between(3, 5),
                source_df["reassignment_count"] >= 6,
            ],
            [
                "0",
                "1-2",
                "3-5",
                "6+",
            ],
            default="INVALID",
        ),
        index=source_df.index,
    )

    comparable = source_df[
        "reassignment_bucket"
    ].notna()

    bucket_mismatch = (
        source_df.loc[
            comparable,
            "reassignment_bucket",
        ]
        != expected_bucket.loc[comparable]
    )

    if bucket_mismatch.any():
        raise AssertionError(
            "Existing reassignment buckets do not match "
            "reassignment_count definitions."
        )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def save_category_report(
    result: pd.DataFrame,
) -> None:
    """
    Save category-level SLA analysis to CSV.
    """

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        CATEGORY_OUTPUT,
        index=False,
    )


def save_priority_report(
    result: pd.DataFrame,
) -> None:
    """
    Save priority-level SLA analysis to CSV.
    """

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        PRIORITY_OUTPUT,
        index=False,
    )


def save_reassignment_report(
    result: pd.DataFrame,
) -> None:
    """
    Save reassignment-level SLA analysis to CSV.
    """

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        REASSIGNMENT_OUTPUT,
        index=False,
    )


# ---------------------------------------------------------------------------
# Terminal summaries
# ---------------------------------------------------------------------------

def print_category_summary(
    result: pd.DataFrame,
) -> None:
    """
    Print a concise category-level operational summary.
    """

    eligible = result[
        result["eligible_for_comparison"]
    ].copy()

    print("\n" + "=" * 70)
    print("CATEGORY SLA RISK ANALYSIS")
    print("=" * 70)

    print(
        f"\nTotal categories analyzed: {len(result)}"
    )

    print(
        "Categories eligible for comparison "
        f"(n >= {MIN_CATEGORY_SAMPLE}): "
        f"{len(eligible)}"
    )

    print("\nTop categories by SLA breach rate:")

    print(
        eligible[
            [
                "category",
                "incident_count",
                "sla_breaches",
                "breach_rate",
                "breach_rate_ci_lower",
                "breach_rate_ci_upper",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print("\nTop categories by number of SLA breaches:")

    print(
        eligible
        .sort_values(
            "sla_breaches",
            ascending=False,
        )[
            [
                "category",
                "incident_count",
                "sla_breaches",
                "breach_rate",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print("\nHighest median resolution time:")

    print(
        eligible
        .sort_values(
            "median_resolution_hours",
            ascending=False,
        )[
            [
                "category",
                "incident_count",
                "median_resolution_hours",
                "p90_resolution_hours",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print("\nReport saved to:")
    print(CATEGORY_OUTPUT)


def print_priority_summary(
    result: pd.DataFrame,
) -> None:
    """
    Print priority-level SLA performance.
    """

    print("\n" + "=" * 70)
    print("PRIORITY SLA RISK ANALYSIS")
    print("=" * 70)

    print("\nSLA performance by priority:")

    print(
        result[
            [
                "priority",
                "incident_count",
                "sla_breaches",
                "breach_rate",
                "breach_rate_ci_lower",
                "breach_rate_ci_upper",
                "median_resolution_hours",
                "p90_resolution_hours",
                "reassignment_rate",
            ]
        ].to_string(
            index=False
        )
    )

    print("\nPriority report saved to:")
    print(PRIORITY_OUTPUT)


def print_reassignment_summary(
    result: pd.DataFrame,
) -> None:
    """
    Print reassignment-level SLA performance.
    """

    print("\n" + "=" * 70)
    print("REASSIGNMENT VS SLA ANALYSIS")
    print("=" * 70)

    print("\nSLA performance by reassignment bucket:")

    print(
        result[
            [
                "reassignment_bucket",
                "incident_count",
                "sla_breaches",
                "breach_rate",
                "breach_rate_ci_lower",
                "breach_rate_ci_upper",
                "median_resolution_hours",
                "p90_resolution_hours",
                "mean_reassignment_count",
            ]
        ].to_string(
            index=False
        )
    )

    print("\nReassignment report saved to:")
    print(REASSIGNMENT_OUTPUT)


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Run Milestone 3 SLA risk analysis.
    """

    print("Loading incident-level dataset...")

    df = load_incident_data()

    print(
        f"Loaded {len(df):,} incident-level records."
    )

    # -----------------------------------------------------------------------
    # Category analysis
    # -----------------------------------------------------------------------

    category_result = analyze_category_sla(
        df
    )

    validate_category_analysis(
        result=category_result,
        source_df=df,
    )

    save_category_report(
        category_result
    )

    print_category_summary(
        category_result
    )

    # -----------------------------------------------------------------------
    # Priority analysis
    # -----------------------------------------------------------------------

    priority_result = analyze_priority_sla(
        df
    )

    validate_priority_analysis(
        result=priority_result,
        source_df=df,
    )

    save_priority_report(
        priority_result
    )

    print_priority_summary(
        priority_result
    )

    # -----------------------------------------------------------------------
    # Reassignment vs SLA analysis
    # -----------------------------------------------------------------------

    reassignment_result = analyze_reassignment_sla(
        df
    )

    validate_reassignment_analysis(
        result=reassignment_result,
        source_df=df,
    )

    save_reassignment_report(
        reassignment_result
    )

    print_reassignment_summary(
        reassignment_result
    )

    # -----------------------------------------------------------------------
    # Final validation status
    # -----------------------------------------------------------------------

    print("\nCategory validation: PASSED")
    print("Priority validation: PASSED")
    print("Reassignment validation: PASSED")


if __name__ == "__main__":
    main()