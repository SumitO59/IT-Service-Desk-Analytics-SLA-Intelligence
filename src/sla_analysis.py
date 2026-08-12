

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
REASSIGNMENT_RESOLUTION_OUTPUT = (
    REPORT_DIR / "reassignment_resolution_performance.csv"
)
ASSIGNMENT_GROUP_OUTPUT = (
    REPORT_DIR / "assignment_group_performance.csv"
)
ASSIGNMENT_GROUP_BOTTLENECK_OUTPUT = (
    REPORT_DIR / "assignment_group_bottleneck_analysis.csv"
)
ASSIGNMENT_GROUP_DRIVER_OUTPUT = (
    REPORT_DIR / "assignment_group_driver_analysis.csv"
)
INCIDENT_SLA_RISK_REFERENCE_OUTPUT = (
    REPORT_DIR / "incident_sla_risk_reference.csv"
)
INCIDENT_SLA_RISK_SCORES_OUTPUT = (
    REPORT_DIR / "incident_sla_risk_scores.csv"
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MIN_CATEGORY_SAMPLE = 100
MIN_ASSIGNMENT_GROUP_SAMPLE = 100
CONFIDENCE_Z = 1.96

# Historical incident-level SLA risk scoring configuration.
SLA_RISK_BASELINE_RATE = 36.58
SLA_RISK_SCORE_MAX = 100.0

SLA_RISK_LOW_THRESHOLD = 25.0
SLA_RISK_MODERATE_THRESHOLD = 50.0
SLA_RISK_HIGH_THRESHOLD = 75.0


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
        "assignment_group",
        "made_sla",
        "sla_breached",
        "resolution_time_hours",
        "reassignment_count",
        "reassignment_bucket",
        "resolved_at",
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
# Incident-level SLA risk reference tables
# ---------------------------------------------------------------------------

def build_incident_sla_risk_reference(
    category_result: pd.DataFrame,
    priority_result: pd.DataFrame,
    reassignment_result: pd.DataFrame,
    assignment_group_result: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build historical SLA-risk reference tables for incident-level scoring.

    The reference rates are derived from previously validated aggregate
    analyses. Category and assignment-group references are restricted to
    populations eligible for primary comparison.

    This function does not calculate an incident-level risk score.
    It only establishes the historical risk reference data that will
    later be mapped onto individual incidents.

    The analysis is descriptive and does not imply causality.
    """

    required_priority_columns = [
        "priority",
        "incident_count",
        "sla_breaches",
        "breach_rate",
        "breach_rate_ci_lower",
        "breach_rate_ci_upper",
    ]

    required_reassignment_columns = [
        "reassignment_bucket",
        "incident_count",
        "sla_breaches",
        "breach_rate",
        "breach_rate_ci_lower",
        "breach_rate_ci_upper",
    ]

    required_category_columns = [
        "category",
        "incident_count",
        "sla_breaches",
        "breach_rate",
        "breach_rate_ci_lower",
        "breach_rate_ci_upper",
        "eligible_for_comparison",
    ]

    required_assignment_group_columns = [
        "assignment_group",
        "incident_count",
        "sla_breaches",
        "breach_rate",
        "breach_rate_ci_lower",
        "breach_rate_ci_upper",
        "eligible_for_comparison",
    ]

    datasets = {
        "priority": (
            priority_result,
            required_priority_columns,
        ),
        "reassignment": (
            reassignment_result,
            required_reassignment_columns,
        ),
        "category": (
            category_result,
            required_category_columns,
        ),
        "assignment_group": (
            assignment_group_result,
            required_assignment_group_columns,
        ),
    }

    for dataset_name, (
        dataset,
        required_columns,
    ) in datasets.items():

        missing_columns = [
            column
            for column in required_columns
            if column not in dataset.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Required {dataset_name} risk-reference columns "
                "missing: "
                + ", ".join(missing_columns)
            )

    category_reference = (
        category_result[
            category_result["eligible_for_comparison"]
        ][
            [
                "category",
                "incident_count",
                "sla_breaches",
                "breach_rate",
                "breach_rate_ci_lower",
                "breach_rate_ci_upper",
            ]
        ]
        .copy()
    )

    category_reference["factor_type"] = "Category"
    category_reference["factor_value"] = (
        category_reference["category"]
    )

    assignment_group_reference = (
        assignment_group_result[
            assignment_group_result["eligible_for_comparison"]
        ][
            [
                "assignment_group",
                "incident_count",
                "sla_breaches",
                "breach_rate",
                "breach_rate_ci_lower",
                "breach_rate_ci_upper",
            ]
        ]
        .copy()
    )

    assignment_group_reference["factor_type"] = (
        "Assignment Group"
    )

    assignment_group_reference["factor_value"] = (
        assignment_group_reference["assignment_group"]
    )

    priority_reference = (
        priority_result[
            [
                "priority",
                "incident_count",
                "sla_breaches",
                "breach_rate",
                "breach_rate_ci_lower",
                "breach_rate_ci_upper",
            ]
        ]
        .copy()
    )

    priority_reference["factor_type"] = "Priority"
    priority_reference["factor_value"] = (
        priority_reference["priority"]
    )

    reassignment_reference = (
        reassignment_result[
            [
                "reassignment_bucket",
                "incident_count",
                "sla_breaches",
                "breach_rate",
                "breach_rate_ci_lower",
                "breach_rate_ci_upper",
            ]
        ]
        .copy()
    )

    reassignment_reference["factor_type"] = (
        "Reassignment Bucket"
    )

    reassignment_reference["factor_value"] = (
        reassignment_reference["reassignment_bucket"]
    )

    reference_tables = [
        priority_reference[
            [
                "factor_type",
                "factor_value",
                "incident_count",
                "sla_breaches",
                "breach_rate",
                "breach_rate_ci_lower",
                "breach_rate_ci_upper",
            ]
        ],
        reassignment_reference[
            [
                "factor_type",
                "factor_value",
                "incident_count",
                "sla_breaches",
                "breach_rate",
                "breach_rate_ci_lower",
                "breach_rate_ci_upper",
            ]
        ],
        category_reference[
            [
                "factor_type",
                "factor_value",
                "incident_count",
                "sla_breaches",
                "breach_rate",
                "breach_rate_ci_lower",
                "breach_rate_ci_upper",
            ]
        ],
        assignment_group_reference[
            [
                "factor_type",
                "factor_value",
                "incident_count",
                "sla_breaches",
                "breach_rate",
                "breach_rate_ci_lower",
                "breach_rate_ci_upper",
            ]
        ],
    ]

    result = (
        pd.concat(
            reference_tables,
            ignore_index=True,
        )
        .sort_values(
            by=[
                "factor_type",
                "breach_rate",
                "incident_count",
            ],
            ascending=[
                True,
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    return result\

# ---------------------------------------------------------------------------
# Incident-level SLA risk scoring
# ---------------------------------------------------------------------------

def build_incident_sla_risk_scores(
    df: pd.DataFrame,
    risk_reference: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a historical, descriptive SLA-risk score for each incident.

    The score combines historical SLA breach rates for priority,
    reassignment bucket, category, and assignment group.

    This is a descriptive historical risk index, not a predicted
    probability and not a machine-learning model.
    """

    required_incident_columns = [
        "number",
        "priority",
        "reassignment_bucket",
        "category",
        "assignment_group",
    ]

    required_reference_columns = [
        "factor_type",
        "factor_value",
        "incident_count",
        "breach_rate",
        "breach_rate_ci_lower",
        "breach_rate_ci_upper",
    ]

    missing_incident_columns = [
        column
        for column in required_incident_columns
        if column not in df.columns
    ]

    if missing_incident_columns:
        raise ValueError(
            "Required incident columns missing from dataset: "
            + ", ".join(missing_incident_columns)
        )

    missing_reference_columns = [
        column
        for column in required_reference_columns
        if column not in risk_reference.columns
    ]

    if missing_reference_columns:
        raise ValueError(
            "Required risk-reference columns missing: "
            + ", ".join(missing_reference_columns)
        )

    if risk_reference.empty:
        raise ValueError(
            "Risk reference table is empty."
        )

    if risk_reference[
        ["factor_type", "factor_value"]
    ].duplicated().any():
        raise ValueError(
            "Risk reference contains duplicate factor-value combinations."
        )

    if risk_reference["breach_rate"].isna().any():
        raise ValueError(
            "Risk reference contains missing breach rates."
        )

    if (
        (risk_reference["breach_rate"] < 0).any()
        or (risk_reference["breach_rate"] > 100).any()
    ):
        raise ValueError(
            "Risk reference contains breach rates outside [0, 100]."
        )

    baseline = SLA_RISK_BASELINE_RATE
    denominator = SLA_RISK_SCORE_MAX - baseline

    if denominator <= 0:
        raise ValueError(
            "SLA risk scoring denominator must be positive."
        )

    result = df[
        [
            "number",
            "priority",
            "reassignment_bucket",
            "category",
            "assignment_group",
        ]
    ].copy()

    factor_definitions = {
        "priority": (
            "Priority",
            "priority_historical_breach_rate",
            "priority_risk_contribution",
        ),
        "reassignment_bucket": (
            "Reassignment Bucket",
            "reassignment_historical_breach_rate",
            "reassignment_risk_contribution",
        ),
        "category": (
            "Category",
            "category_historical_breach_rate",
            "category_risk_contribution",
        ),
        "assignment_group": (
            "Assignment Group",
            "assignment_group_historical_breach_rate",
            "assignment_group_risk_contribution",
        ),
    }

    contribution_columns = []

    for incident_column, (
        reference_type,
        rate_column,
        contribution_column,
    ) in factor_definitions.items():

        reference_subset = risk_reference[
            risk_reference["factor_type"] == reference_type
        ][
            [
                "factor_value",
                "breach_rate",
            ]
        ].copy()

        reference_subset = reference_subset.rename(
            columns={
                "factor_value": incident_column,
                "breach_rate": rate_column,
            }
        )

        result = result.merge(
            reference_subset,
            on=incident_column,
            how="left",
            validate="many_to_one",
        )

        result[contribution_column] = (
            (
                result[rate_column] - baseline
            ) / denominator
        ).clip(
            lower=0.0,
            upper=1.0,
        )

        result.loc[
            result[rate_column].isna(),
            contribution_column,
        ] = np.nan

        contribution_columns.append(
            contribution_column
        )

    result["risk_factor_count"] = (
        result[contribution_columns]
        .notna()
        .sum(axis=1)
    )

    result["sla_risk_score"] = (
        result[contribution_columns]
        .mean(axis=1, skipna=True)
        * SLA_RISK_SCORE_MAX
    )

    result["sla_risk_band"] = pd.cut(
        result["sla_risk_score"],
        bins=[
            -np.inf,
            SLA_RISK_LOW_THRESHOLD,
            SLA_RISK_MODERATE_THRESHOLD,
            SLA_RISK_HIGH_THRESHOLD,
            np.inf,
        ],
        labels=[
            "Low",
            "Moderate",
            "High",
            "Critical",
        ],
        right=False,
    )

    return result


# ---------------------------------------------------------------------------
# Incident-level SLA risk scoring validation
# ---------------------------------------------------------------------------

def validate_incident_sla_risk_scores(
    result: pd.DataFrame,
    source_df: pd.DataFrame,
    risk_reference: pd.DataFrame,
) -> None:
    """
    Validate incident-level historical SLA-risk scores.

    The validator checks row preservation, incident identity, factor
    coverage, contribution bounds, score bounds, risk-band validity,
    mathematical score reconstruction, and historical reference mapping.

    This validation concerns a descriptive historical risk index and does
    not validate predictive model performance.
    """

    required_result_columns = [
        "number",
        "priority",
        "reassignment_bucket",
        "category",
        "assignment_group",
        "priority_historical_breach_rate",
        "reassignment_historical_breach_rate",
        "category_historical_breach_rate",
        "assignment_group_historical_breach_rate",
        "priority_risk_contribution",
        "reassignment_risk_contribution",
        "category_risk_contribution",
        "assignment_group_risk_contribution",
        "risk_factor_count",
        "sla_risk_score",
        "sla_risk_band",
    ]

    missing_result_columns = [
        column
        for column in required_result_columns
        if column not in result.columns
    ]

    if missing_result_columns:
        raise AssertionError(
            "Required SLA-risk score columns missing: "
            + ", ".join(missing_result_columns)
        )

    if len(result) != len(source_df):
        raise AssertionError(
            "Incident-level SLA-risk scoring changed the source row count."
        )

    if result["number"].nunique() != source_df["number"].nunique():
        raise AssertionError(
            "Incident-level SLA-risk scoring changed incident identity."
        )

    if result["number"].duplicated().any():
        raise AssertionError(
            "Duplicate incident IDs detected in SLA-risk scoring output."
        )

    contribution_columns = [
        "priority_risk_contribution",
        "reassignment_risk_contribution",
        "category_risk_contribution",
        "assignment_group_risk_contribution",
    ]

    historical_rate_columns = [
        "priority_historical_breach_rate",
        "reassignment_historical_breach_rate",
        "category_historical_breach_rate",
        "assignment_group_historical_breach_rate",
    ]

    # -----------------------------------------------------------------------
    # Factor coverage
    # -----------------------------------------------------------------------

    expected_factor_count = (
        result[historical_rate_columns]
        .notna()
        .sum(axis=1)
    )

    if not (
        result["risk_factor_count"].to_numpy()
        == expected_factor_count.to_numpy()
    ).all():
        raise AssertionError(
            "Risk-factor count is inconsistent with historical "
            "reference coverage."
        )

    if (
        result["risk_factor_count"] < 2
    ).any() or (
        result["risk_factor_count"] > 4
    ).any():
        raise AssertionError(
            "Risk-factor count must be between 2 and 4."
        )

    # -----------------------------------------------------------------------
    # Historical rate validation
    # -----------------------------------------------------------------------

    for column in historical_rate_columns:
        valid_rates = result[column].dropna()

        if (
            (valid_rates < 0).any()
            or (valid_rates > 100).any()
        ):
            raise AssertionError(
                f"Historical breach rates outside [0, 100] detected "
                f"in {column}."
            )

    # -----------------------------------------------------------------------
    # Contribution validation
    # -----------------------------------------------------------------------

    for column in contribution_columns:
        valid_contributions = result[column].dropna()

        if (
            (valid_contributions < 0).any()
            or (valid_contributions > 1).any()
        ):
            raise AssertionError(
                f"Risk contributions outside [0, 1] detected in {column}."
            )

    contribution_missingness = (
        result[contribution_columns].isna().to_numpy()
    )

    rate_missingness = (
        result[historical_rate_columns].isna().to_numpy()
    )

    if not (
        contribution_missingness == rate_missingness
    ).all():
        raise AssertionError(
            "Risk contribution missingness does not match "
            "historical-rate coverage."
        )

    # -----------------------------------------------------------------------
    # Score validation
    # -----------------------------------------------------------------------

    if result["sla_risk_score"].isna().any():
        raise AssertionError(
            "Missing SLA risk scores detected."
        )

    if (
        result["sla_risk_score"] < 0
    ).any() or (
        result["sla_risk_score"] > SLA_RISK_SCORE_MAX
    ).any():
        raise AssertionError(
            "SLA risk scores outside [0, 100] detected."
        )

    expected_score = (
        result[contribution_columns]
        .mean(axis=1, skipna=True)
        * SLA_RISK_SCORE_MAX
    )

    score_error = (
        result["sla_risk_score"] - expected_score
    ).abs()

    if score_error.max() > 1e-10:
        raise AssertionError(
            "SLA risk score does not match its mathematical definition."
        )

    # -----------------------------------------------------------------------
    # Risk-band validation
    # -----------------------------------------------------------------------

    valid_bands = {
        "Low",
        "Moderate",
        "High",
        "Critical",
    }

    if result["sla_risk_band"].isna().any():
        raise AssertionError(
            "Missing SLA risk bands detected."
        )

    actual_bands = set(
        result["sla_risk_band"]
        .astype(str)
        .unique()
    )

    unexpected_bands = actual_bands - valid_bands

    if unexpected_bands:
        raise AssertionError(
            "Unexpected SLA risk bands detected: "
            + ", ".join(sorted(unexpected_bands))
        )

    expected_bands = pd.cut(
        result["sla_risk_score"],
        bins=[
            -np.inf,
            SLA_RISK_LOW_THRESHOLD,
            SLA_RISK_MODERATE_THRESHOLD,
            SLA_RISK_HIGH_THRESHOLD,
            np.inf,
        ],
        labels=[
            "Low",
            "Moderate",
            "High",
            "Critical",
        ],
        right=False,
    )

    if not (
        result["sla_risk_band"].astype(str).to_numpy()
        == expected_bands.astype(str).to_numpy()
    ).all():
        raise AssertionError(
            "SLA risk bands are inconsistent with configured thresholds."
        )

    # -----------------------------------------------------------------------
    # Historical reference mapping validation
    # -----------------------------------------------------------------------

    mapping_definitions = [
        (
            "priority",
            "Priority",
            "priority_historical_breach_rate",
        ),
        (
            "reassignment_bucket",
            "Reassignment Bucket",
            "reassignment_historical_breach_rate",
        ),
        (
            "category",
            "Category",
            "category_historical_breach_rate",
        ),
        (
            "assignment_group",
            "Assignment Group",
            "assignment_group_historical_breach_rate",
        ),
    ]

    for (
        incident_column,
        reference_type,
        rate_column,
    ) in mapping_definitions:

        reference_subset = risk_reference[
            risk_reference["factor_type"] == reference_type
        ][
            [
                "factor_value",
                "breach_rate",
            ]
        ]

        expected_mapping = dict(
            zip(
                reference_subset["factor_value"],
                reference_subset["breach_rate"],
            )
        )

        for _, row in result[
            [
                incident_column,
                rate_column,
            ]
        ].drop_duplicates().iterrows():

            factor_value = row[incident_column]
            actual_rate = row[rate_column]

            if factor_value not in expected_mapping:
                if not pd.isna(actual_rate):
                    raise AssertionError(
                        f"Unexpected historical reference mapping for "
                        f"{reference_type}: {factor_value}"
                    )
                continue

            expected_rate = expected_mapping[factor_value]

            if pd.isna(actual_rate):
                raise AssertionError(
                    f"Missing historical reference mapping for "
                    f"{reference_type}: {factor_value}"
                )

            if abs(actual_rate - expected_rate) > 1e-10:
                raise AssertionError(
                    f"Historical reference mapping mismatch for "
                    f"{reference_type}: {factor_value}"
                )


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
# Reassignment vs resolution-time analysis
# ---------------------------------------------------------------------------

def analyze_reassignment_resolution(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Analyze observed resolution-time performance across reassignment
    buckets.

    Resolution-time statistics are calculated only for incidents with a
    non-null resolution duration.

    Resolution coverage is reported separately because missing resolution
    timestamps are not distributed uniformly across reassignment buckets.
    """

    reassignment_df = df.copy()

    grouped = (
        reassignment_df
        .groupby(
            "reassignment_bucket",
            observed=True,
        )
        .agg(
            incident_count=("number", "nunique"),
            resolved_incidents=(
                "resolved_at",
                "count",
            ),
            missing_resolution=(
                "resolved_at",
                lambda x: x.isna().sum(),
            ),
            median_resolution_hours=(
                "resolution_time_hours",
                "median",
            ),
            p90_resolution_hours=(
                "resolution_time_hours",
                lambda x: x.quantile(0.90),
            ),
            mean_resolution_hours=(
                "resolution_time_hours",
                "mean",
            ),
        )
        .reset_index()
    )

    grouped["resolution_coverage"] = (
        grouped["resolved_incidents"]
        / grouped["incident_count"]
        * 100
    )

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
# Assignment-group performance analysis
# ---------------------------------------------------------------------------

def analyze_assignment_group_performance(
    df: pd.DataFrame,
    min_sample: int = MIN_ASSIGNMENT_GROUP_SAMPLE,
) -> pd.DataFrame:
    """
    Calculate SLA and resolution performance by assignment group.

    Assignment groups with fewer than `min_sample` incidents are retained
    in the output but excluded from primary performance comparisons.

    Incidents with missing assignment groups are excluded from group-level
    rankings and reported separately.
    """

    assignment_df = df.copy()

    assignment_df = assignment_df.dropna(
        subset=["assignment_group"]
    )

    grouped = (
        assignment_df
        .groupby(
            "assignment_group",
            observed=True,
        )
        .agg(
            incident_count=("number", "nunique"),
            resolved_incidents=(
                "resolved_at",
                "count",
            ),
            missing_resolution=(
                "resolved_at",
                lambda x: x.isna().sum(),
            ),
            sla_breaches=(
                "sla_breached",
                "sum",
            ),
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

    grouped["resolution_coverage"] = (
        grouped["resolved_incidents"]
        / grouped["incident_count"]
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
        "resolution_coverage",
        "reassignment_rate",
        "breach_rate_ci_lower",
        "breach_rate_ci_upper",
    ]

    for column in percentage_columns:
        grouped[column] = grouped[column] * 100

    grouped = (
        grouped
        .sort_values(
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
        )
        .reset_index(drop=True)
    )

    return grouped

# ---------------------------------------------------------------------------
# Assignment-group bottleneck analysis
# ---------------------------------------------------------------------------

def analyze_assignment_group_bottlenecks(
    assignment_group_result: pd.DataFrame,
) -> pd.DataFrame:
    """
    Classify eligible assignment groups by operational volume and SLA risk.

    Classification is based on the median incident volume and median SLA
    breach rate among assignment groups eligible for primary comparison.

    The analysis is descriptive and intended for operational prioritization.
    It does not imply causal relationships.
    """

    required_columns = [
        "assignment_group",
        "incident_count",
        "sla_breaches",
        "breach_rate",
        "median_resolution_hours",
        "p90_resolution_hours",
        "eligible_for_comparison",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in assignment_group_result.columns
    ]

    if missing_columns:
        raise ValueError(
            "Required assignment-group columns missing: "
            + ", ".join(missing_columns)
        )

    eligible = assignment_group_result[
        assignment_group_result["eligible_for_comparison"]
    ].copy()

    if eligible.empty:
        raise ValueError(
            "No assignment groups are eligible for bottleneck analysis."
        )

    volume_threshold = eligible["incident_count"].median()
    breach_rate_threshold = eligible["breach_rate"].median()

    eligible["volume_class"] = np.where(
        eligible["incident_count"] >= volume_threshold,
        "High Volume",
        "Lower Volume",
    )

    eligible["breach_risk_class"] = np.where(
        eligible["breach_rate"] >= breach_rate_threshold,
        "High Breach Risk",
        "Lower Breach Risk",
    )

    eligible["bottleneck_class"] = np.select(
        [
            (
                (eligible["incident_count"] >= volume_threshold)
                & (eligible["breach_rate"] >= breach_rate_threshold)
            ),
            (
                (eligible["incident_count"] >= volume_threshold)
                & (eligible["breach_rate"] < breach_rate_threshold)
            ),
            (
                (eligible["incident_count"] < volume_threshold)
                & (eligible["breach_rate"] >= breach_rate_threshold)
            ),
        ],
        [
            "Critical Bottleneck",
            "Volume Bottleneck",
            "SLA Risk",
        ],
        default="Lower Priority",
    )

    eligible["volume_threshold"] = volume_threshold
    eligible["breach_rate_threshold"] = breach_rate_threshold

    output_columns = [
        "assignment_group",
        "incident_count",
        "sla_breaches",
        "breach_rate",
        "median_resolution_hours",
        "p90_resolution_hours",
        "volume_class",
        "breach_risk_class",
        "bottleneck_class",
        "volume_threshold",
        "breach_rate_threshold",
    ]

    result = (
        eligible[output_columns]
        .sort_values(
            by=[
                "bottleneck_class",
                "breach_rate",
                "incident_count",
            ],
            ascending=[
                True,
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    return result

# ---------------------------------------------------------------------------
# Assignment-group operational driver analysis
# ---------------------------------------------------------------------------

def analyze_assignment_group_drivers(
    assignment_group_result: pd.DataFrame,
    assignment_group_bottleneck_result: pd.DataFrame,
) -> pd.DataFrame:
    """
    Analyze operational characteristics associated with assignment-group
    bottleneck classification.

    The analysis uses only assignment groups eligible for primary comparison.
    It compares Critical Bottleneck groups against all other eligible groups
    using reassignment and resolution-time metrics.

    This is descriptive diagnostic analysis and does not imply causality.
    """

    required_group_columns = [
        "assignment_group",
        "incident_count",
        "sla_breaches",
        "breach_rate",
        "median_resolution_hours",
        "p90_resolution_hours",
        "reassignment_rate",
        "mean_reassignment_count",
    ]

    missing_group_columns = [
        column
        for column in required_group_columns
        if column not in assignment_group_result.columns
    ]

    if missing_group_columns:
        raise ValueError(
            "Required assignment-group driver columns missing: "
            + ", ".join(missing_group_columns)
        )

    required_bottleneck_columns = [
        "assignment_group",
        "bottleneck_class",
    ]

    missing_bottleneck_columns = [
        column
        for column in required_bottleneck_columns
        if column not in assignment_group_bottleneck_result.columns
    ]

    if missing_bottleneck_columns:
        raise ValueError(
            "Required bottleneck columns missing: "
            + ", ".join(missing_bottleneck_columns)
        )

    eligible = assignment_group_result[
        assignment_group_result["eligible_for_comparison"]
    ].copy()

    if eligible.empty:
        raise ValueError(
            "No assignment groups are eligible for driver analysis."
        )

    bottleneck_classes = (
        assignment_group_bottleneck_result[
            [
                "assignment_group",
                "bottleneck_class",
            ]
        ]
        .drop_duplicates(
            subset=["assignment_group"]
        )
    )

    result = eligible.merge(
        bottleneck_classes,
        on="assignment_group",
        how="left",
        validate="one_to_one",
    )

    if result["bottleneck_class"].isna().any():
        raise AssertionError(
            "Some eligible assignment groups are missing "
            "bottleneck classifications."
        )

    result["driver_segment"] = np.where(
        result["bottleneck_class"]
        == "Critical Bottleneck",
        "Critical Bottleneck",
        "Other Eligible Groups",
    )

    output_columns = [
        "assignment_group",
        "bottleneck_class",
        "driver_segment",
        "incident_count",
        "sla_breaches",
        "breach_rate",
        "median_resolution_hours",
        "p90_resolution_hours",
        "reassignment_rate",
        "mean_reassignment_count",
    ]

    return (
        result[output_columns]
        .sort_values(
            by=[
                "driver_segment",
                "breach_rate",
                "incident_count",
            ],
            ascending=[
                True,
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )

# ---------------------------------------------------------------------------
# Assignment-group driver validation
# ---------------------------------------------------------------------------

def validate_assignment_group_drivers(
    result: pd.DataFrame,
    assignment_group_result: pd.DataFrame,
    assignment_group_bottleneck_result: pd.DataFrame,
) -> None:
    """
    Validate assignment-group operational driver analysis.
    """

    required_columns = [
        "assignment_group",
        "bottleneck_class",
        "driver_segment",
        "incident_count",
        "sla_breaches",
        "breach_rate",
        "median_resolution_hours",
        "p90_resolution_hours",
        "reassignment_rate",
        "mean_reassignment_count",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in result.columns
    ]

    if missing_columns:
        raise AssertionError(
            "Required assignment-group driver columns missing: "
            + ", ".join(missing_columns)
        )

    eligible = assignment_group_result[
        assignment_group_result["eligible_for_comparison"]
    ]

    if len(result) != len(eligible):
        raise AssertionError(
            "Driver result count does not match the number "
            "of eligible assignment groups: "
            f"result={len(result)}, eligible={len(eligible)}"
        )

    if result["assignment_group"].nunique() != len(result):
        raise AssertionError(
            "Duplicate assignment groups detected in driver result."
        )

    if result[required_columns].isna().any().any():
        raise AssertionError(
            "Missing values detected in assignment-group driver result."
        )

    expected_classes = set(
        assignment_group_bottleneck_result[
            "bottleneck_class"
        ].unique()
    )

    actual_classes = set(
        result["bottleneck_class"].unique()
    )

    if actual_classes != expected_classes:
        raise AssertionError(
            "Bottleneck classifications do not reconcile "
            "with the Milestone 3.6 result."
        )

    expected_driver_segment = np.where(
        result["bottleneck_class"]
        == "Critical Bottleneck",
        "Critical Bottleneck",
        "Other Eligible Groups",
    )

    if not (
        result["driver_segment"].to_numpy()
        == expected_driver_segment
    ).all():
        raise AssertionError(
            "Driver-segment classification is inconsistent."
        )

    source_lookup = (
        eligible[
            [
                "assignment_group",
                "incident_count",
                "sla_breaches",
                "breach_rate",
                "median_resolution_hours",
                "p90_resolution_hours",
                "reassignment_rate",
                "mean_reassignment_count",
            ]
        ]
        .set_index("assignment_group")
        .sort_index()
    )

    result_lookup = (
        result[
            [
                "assignment_group",
                "incident_count",
                "sla_breaches",
                "breach_rate",
                "median_resolution_hours",
                "p90_resolution_hours",
                "reassignment_rate",
                "mean_reassignment_count",
            ]
        ]
        .set_index("assignment_group")
        .sort_index()
    )

    if not np.allclose(
        source_lookup.to_numpy(),
        result_lookup.to_numpy(),
        equal_nan=True,
    ):
        raise AssertionError(
            "Operational driver metrics do not reconcile "
            "with the validated assignment-group analysis."
        )

    critical_count = (
        result["driver_segment"]
        == "Critical Bottleneck"
    ).sum()

    if critical_count != (
        assignment_group_bottleneck_result[
            "bottleneck_class"
        ]
        == "Critical Bottleneck"
    ).sum():
        raise AssertionError(
            "Critical-bottleneck count does not reconcile "
            "with Milestone 3.6."
        )
# ---------------------------------------------------------------------------
# Assignment-group bottleneck validation
# ---------------------------------------------------------------------------

def validate_assignment_group_bottlenecks(
    result: pd.DataFrame,
    assignment_group_result: pd.DataFrame,
) -> None:
    """
    Validate assignment-group bottleneck classification.

    The validator ensures that only eligible assignment groups are classified,
    that all four classification dimensions are internally consistent, and
    that the classification counts reconcile with the eligible population.
    """

    required_columns = [
        "assignment_group",
        "incident_count",
        "breach_rate",
        "volume_class",
        "breach_risk_class",
        "bottleneck_class",
        "volume_threshold",
        "breach_rate_threshold",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in result.columns
    ]

    if missing_columns:
        raise AssertionError(
            "Required bottleneck columns missing: "
            + ", ".join(missing_columns)
        )

    eligible_count = (
        assignment_group_result["eligible_for_comparison"]
        .sum()
    )

    if len(result) != eligible_count:
        raise AssertionError(
            "Bottleneck result count does not match the number "
            "of eligible assignment groups: "
            f"result={len(result)}, eligible={eligible_count}"
        )

    if result["assignment_group"].nunique() != len(result):
        raise AssertionError(
            "Duplicate assignment groups detected in bottleneck result."
        )

    if result[required_columns].isna().any().any():
        raise AssertionError(
            "Missing values detected in required bottleneck fields."
        )

    volume_threshold = result["volume_threshold"].iloc[0]
    breach_rate_threshold = result["breach_rate_threshold"].iloc[0]

    expected_volume_class = np.where(
        result["incident_count"] >= volume_threshold,
        "High Volume",
        "Lower Volume",
    )

    if not (
        result["volume_class"].to_numpy()
        == expected_volume_class
    ).all():
        raise AssertionError(
            "Assignment-group volume classification is inconsistent "
            "with the volume threshold."
        )

    expected_breach_risk_class = np.where(
        result["breach_rate"] >= breach_rate_threshold,
        "High Breach Risk",
        "Lower Breach Risk",
    )

    if not (
        result["breach_risk_class"].to_numpy()
        == expected_breach_risk_class
    ).all():
        raise AssertionError(
            "Assignment-group breach-risk classification is inconsistent "
            "with the breach-rate threshold."
        )

    expected_bottleneck_class = np.select(
        [
            (
                (result["incident_count"] >= volume_threshold)
                & (result["breach_rate"] >= breach_rate_threshold)
            ),
            (
                (result["incident_count"] >= volume_threshold)
                & (result["breach_rate"] < breach_rate_threshold)
            ),
            (
                (result["incident_count"] < volume_threshold)
                & (result["breach_rate"] >= breach_rate_threshold)
            ),
        ],
        [
            "Critical Bottleneck",
            "Volume Bottleneck",
            "SLA Risk",
        ],
        default="Lower Priority",
    )

    if not (
        result["bottleneck_class"].to_numpy()
        == expected_bottleneck_class
    ).all():
        raise AssertionError(
            "Assignment-group bottleneck classification is inconsistent "
            "with the volume and breach-rate thresholds."
        )

    valid_classes = {
        "Critical Bottleneck",
        "Volume Bottleneck",
        "SLA Risk",
        "Lower Priority",
    }

    unexpected_classes = set(
        result["bottleneck_class"].unique()
    ) - valid_classes

    if unexpected_classes:
        raise AssertionError(
            "Unexpected bottleneck classifications detected: "
            + ", ".join(sorted(unexpected_classes))
        )

    threshold_values = result[
        [
            "volume_threshold",
            "breach_rate_threshold",
        ]
    ].nunique()

    if (
        threshold_values["volume_threshold"] != 1
        or threshold_values["breach_rate_threshold"] != 1
    ):
        raise AssertionError(
            "Bottleneck thresholds are not consistent across results."
        )


# ---------------------------------------------------------------------------
# Incident-level SLA risk reference validation
# ---------------------------------------------------------------------------

def validate_incident_sla_risk_reference(
    result: pd.DataFrame,
    category_result: pd.DataFrame,
    priority_result: pd.DataFrame,
    reassignment_result: pd.DataFrame,
    assignment_group_result: pd.DataFrame,
) -> None:
    """
    Validate the historical SLA-risk reference tables.

    The validator ensures that the reference population reconciles with
    the previously validated aggregate analyses and that all historical
    breach-rate estimates have valid confidence intervals.
    """

    required_columns = [
        "factor_type",
        "factor_value",
        "incident_count",
        "sla_breaches",
        "breach_rate",
        "breach_rate_ci_lower",
        "breach_rate_ci_upper",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in result.columns
    ]

    if missing_columns:
        raise AssertionError(
            "Required incident SLA risk reference columns missing: "
            + ", ".join(missing_columns)
        )

    if result[required_columns].isna().any().any():
        raise AssertionError(
            "Missing values detected in incident SLA risk references."
        )

    expected_factor_types = {
        "Priority",
        "Reassignment Bucket",
        "Category",
        "Assignment Group",
    }

    actual_factor_types = set(
        result["factor_type"].unique()
    )

    if actual_factor_types != expected_factor_types:
        raise AssertionError(
            "Unexpected incident SLA risk factor types: "
            f"{actual_factor_types}"
        )

    expected_counts = {
        "Priority": len(priority_result),
        "Reassignment Bucket": len(reassignment_result),
        "Category": int(
            category_result[
                category_result["eligible_for_comparison"]
            ].shape[0]
        ),
        "Assignment Group": int(
            assignment_group_result[
                assignment_group_result["eligible_for_comparison"]
            ].shape[0]
        ),
    }

    actual_counts = (
        result["factor_type"]
        .value_counts()
        .to_dict()
    )

    if actual_counts != expected_counts:
        raise AssertionError(
            "Incident SLA risk reference counts do not reconcile: "
            f"expected={expected_counts}, "
            f"actual={actual_counts}"
        )

    if result.duplicated(
        subset=[
            "factor_type",
            "factor_value",
        ]
    ).any():
        raise AssertionError(
            "Duplicate incident SLA risk reference factors detected."
        )

    if (
        result["incident_count"] <= 0
    ).any():
        raise AssertionError(
            "Invalid non-positive incident counts detected."
        )

    if (
        result["sla_breaches"] < 0
    ).any():
        raise AssertionError(
            "Negative SLA breach counts detected."
        )

    if (
        result["sla_breaches"]
        > result["incident_count"]
    ).any():
        raise AssertionError(
            "SLA breach counts exceed incident counts."
        )

    expected_breach_rate = (
        result["sla_breaches"]
        / result["incident_count"]
        * 100
    )

    if not np.isclose(
        result["breach_rate"],
        expected_breach_rate,
    ).all():
        raise AssertionError(
            "Incident SLA risk breach-rate calculations "
            "are inconsistent."
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
            "Some incident SLA risk breach rates fall "
            "outside their confidence intervals."
        )

    if (
        result["breach_rate_ci_lower"] < 0
    ).any() or (
        result["breach_rate_ci_upper"] > 100
    ).any():
        raise AssertionError(
            "Invalid incident SLA risk Wilson confidence intervals."
        )


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
# Reassignment resolution validation
# ---------------------------------------------------------------------------

def validate_reassignment_resolution_analysis(
    result: pd.DataFrame,
    source_df: pd.DataFrame,
) -> None:
    """
    Validate reassignment-level resolution-time aggregation.
    """

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
            "Resolution analysis incident-count mismatch: "
            f"source={source_incidents}, "
            f"result={result_incidents}"
        )

    source_resolved = (
        source_df.loc[
            source_df["reassignment_bucket"].notna(),
            "resolved_at",
        ]
        .notna()
        .sum()
    )

    result_resolved = (
        result["resolved_incidents"].sum()
    )

    if source_resolved != result_resolved:
        raise AssertionError(
            "Resolution analysis resolved-count mismatch: "
            f"source={source_resolved}, "
            f"result={result_resolved}"
        )

    source_missing = (
        source_df.loc[
            source_df["reassignment_bucket"].notna(),
            "resolved_at",
        ]
        .isna()
        .sum()
    )

    result_missing = (
        result["missing_resolution"].sum()
    )

    if source_missing != result_missing:
        raise AssertionError(
            "Resolution analysis missing-count mismatch: "
            f"source={source_missing}, "
            f"result={result_missing}"
        )

    expected_coverage = (
        result["resolved_incidents"]
        / result["incident_count"]
        * 100
    )

    if not np.isclose(
        result["resolution_coverage"],
        expected_coverage,
    ).all():
        raise AssertionError(
            "Resolution coverage calculation is inconsistent."
        )

    if (
        result["median_resolution_hours"] < 0
    ).any():
        raise AssertionError(
            "Negative median resolution time detected."
        )

    if (
        result["p90_resolution_hours"] < 0
    ).any():
        raise AssertionError(
            "Negative P90 resolution time detected."
        )

    if (
        result["mean_resolution_hours"] < 0
    ).any():
        raise AssertionError(
            "Negative mean resolution time detected."
        )

    if (
        result["resolution_coverage"] < 0
    ).any() or (
        result["resolution_coverage"] > 100
    ).any():
        raise AssertionError(
            "Invalid resolution coverage detected."
        )


# ---------------------------------------------------------------------------
# Assignment-group validation
# ---------------------------------------------------------------------------

def validate_assignment_group_analysis(
    result: pd.DataFrame,
    source_df: pd.DataFrame,
) -> None:
    """
    Validate assignment-group performance aggregation.

    Missing assignment groups are intentionally excluded from the grouped
    result and validated separately.
    """

    valid_assignment_groups = (
        source_df["assignment_group"]
        .dropna()
        .nunique()
    )

    result_assignment_groups = (
        result["assignment_group"].nunique()
    )

    if valid_assignment_groups != result_assignment_groups:
        raise AssertionError(
            "Assignment-group count mismatch: "
            f"source={valid_assignment_groups}, "
            f"result={result_assignment_groups}"
        )

    valid_mask = (
        source_df["assignment_group"].notna()
    )

    source_incidents = (
        valid_mask.sum()
    )

    result_incidents = (
        result["incident_count"].sum()
    )

    if source_incidents != result_incidents:
        raise AssertionError(
            "Assignment-group incident-count mismatch: "
            f"source={source_incidents}, "
            f"result={result_incidents}"
        )

    source_breaches = (
        source_df.loc[
            valid_mask,
            "sla_breached",
        ].sum()
    )

    result_breaches = (
        result["sla_breaches"].sum()
    )

    if source_breaches != result_breaches:
        raise AssertionError(
            "Assignment-group SLA-breach mismatch: "
            f"source={source_breaches}, "
            f"result={result_breaches}"
        )

    source_resolved = (
        source_df.loc[
            valid_mask,
            "resolved_at",
        ]
        .notna()
        .sum()
    )

    result_resolved = (
        result["resolved_incidents"].sum()
    )

    if source_resolved != result_resolved:
        raise AssertionError(
            "Assignment-group resolved-count mismatch: "
            f"source={source_resolved}, "
            f"result={result_resolved}"
        )

    source_missing_resolution = (
        source_df.loc[
            valid_mask,
            "resolved_at",
        ]
        .isna()
        .sum()
    )

    result_missing_resolution = (
        result["missing_resolution"].sum()
    )

    if source_missing_resolution != result_missing_resolution:
        raise AssertionError(
            "Assignment-group missing-resolution mismatch: "
            f"source={source_missing_resolution}, "
            f"result={result_missing_resolution}"
        )

    missing_assignment_groups = (
        source_df["assignment_group"].isna().sum()
    )

    expected_missing_assignment_groups = (
        len(source_df) - source_incidents
    )

    if (
        missing_assignment_groups
        != expected_missing_assignment_groups
    ):
        raise AssertionError(
            "Missing assignment-group count is inconsistent."
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
            "Some assignment-group breach rates fall "
            "outside their confidence intervals."
        )

    if (
        (result["breach_rate_ci_lower"] < 0).any()
        or
        (result["breach_rate_ci_upper"] > 100).any()
    ):
        raise AssertionError(
            "Invalid assignment-group Wilson confidence intervals."
        )

    expected_eligibility = (
        result["incident_count"]
        >= MIN_ASSIGNMENT_GROUP_SAMPLE
    )

    if not (
        result["eligible_for_comparison"]
        == expected_eligibility
    ).all():
        raise AssertionError(
            "Assignment-group sample-size eligibility flag "
            "is inconsistent."
        )

    expected_coverage = (
        result["resolved_incidents"]
        / result["incident_count"]
        * 100
    )

    if not np.isclose(
        result["resolution_coverage"],
        expected_coverage,
    ).all():
        raise AssertionError(
            "Assignment-group resolution coverage "
            "calculation is inconsistent."
        )

    if (
        result["median_resolution_hours"] < 0
    ).any():
        raise AssertionError(
            "Negative assignment-group median resolution "
            "time detected."
        )

    if (
        result["p90_resolution_hours"] < 0
    ).any():
        raise AssertionError(
            "Negative assignment-group P90 resolution "
            "time detected."
        )

    if (
        result["resolution_coverage"] < 0
    ).any() or (
        result["resolution_coverage"] > 100
    ).any():
        raise AssertionError(
            "Invalid assignment-group resolution coverage."
        )

def save_incident_sla_risk_reference_report(
    result: pd.DataFrame,
) -> None:
    """
    Save historical incident SLA-risk reference tables to CSV.
    """

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        INCIDENT_SLA_RISK_REFERENCE_OUTPUT,
        index=False,
    )


def save_incident_sla_risk_scores_report(
    result: pd.DataFrame,
) -> None:
    """
    Save incident-level historical SLA-risk scores to CSV.
    """

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        INCIDENT_SLA_RISK_SCORES_OUTPUT,
        index=False,
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


def save_reassignment_resolution_report(
    result: pd.DataFrame,
) -> None:
    """
    Save reassignment-level resolution-time analysis to CSV.
    """

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        REASSIGNMENT_RESOLUTION_OUTPUT,
        index=False,
    )


def save_assignment_group_report(
    result: pd.DataFrame,
) -> None:
    """
    Save assignment-group performance analysis to CSV.
    """

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        ASSIGNMENT_GROUP_OUTPUT,
        index=False,
    )

def save_assignment_group_bottleneck_report(
    result: pd.DataFrame,
) -> None:
    """
    Save assignment-group bottleneck classification to CSV.
    """

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        ASSIGNMENT_GROUP_BOTTLENECK_OUTPUT,
        index=False,
    )



def save_assignment_group_driver_report(
    result: pd.DataFrame,
) -> None:
    """
    Save assignment-group operational driver analysis to CSV.
    """

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        ASSIGNMENT_GROUP_DRIVER_OUTPUT,
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


def print_reassignment_resolution_summary(
    result: pd.DataFrame,
) -> None:
    """
    Print reassignment-level resolution-time performance.
    """

    print("\n" + "=" * 70)
    print("REASSIGNMENT VS RESOLUTION TIME")
    print("=" * 70)

    print(
        "\nResolution-time performance by reassignment bucket:"
    )

    print(
        result[
            [
                "reassignment_bucket",
                "incident_count",
                "resolved_incidents",
                "missing_resolution",
                "resolution_coverage",
                "median_resolution_hours",
                "p90_resolution_hours",
                "mean_resolution_hours",
            ]
        ].to_string(
            index=False
        )
    )

    print("\nResolution report saved to:")
    print(REASSIGNMENT_RESOLUTION_OUTPUT)


def print_assignment_group_summary(
    result: pd.DataFrame,
    source_df: pd.DataFrame,
) -> None:
    """
    Print assignment-group SLA and resolution performance.

    Breach rate and breach volume are intentionally reported separately.
    """

    eligible = result[
        result["eligible_for_comparison"]
    ].copy()

    missing_assignment_groups = (
        source_df["assignment_group"].isna().sum()
    )

    missing_assignment_group_rate = (
        missing_assignment_groups
        / len(source_df)
        * 100
    )

    print("\n" + "=" * 70)
    print("ASSIGNMENT-GROUP PERFORMANCE")
    print("=" * 70)

    print(
        f"\nTotal assignment groups: {len(result)}"
    )

    print(
        f"Groups eligible for comparison "
        f"(n >= {MIN_ASSIGNMENT_GROUP_SAMPLE}): "
        f"{len(eligible)}"
    )

    print("\nMissing assignment groups:")

    print(
        f"Count: {missing_assignment_groups:,}"
    )

    print(
        f"Percentage: "
        f"{missing_assignment_group_rate:.2f}%"
    )

    print(
        "\nTop assignment groups by SLA breach rate:"
    )

    print(
        eligible[
            [
                "assignment_group",
                "incident_count",
                "sla_breaches",
                "breach_rate",
                "breach_rate_ci_lower",
                "breach_rate_ci_upper",
            ]
        ]
        .sort_values(
            "breach_rate",
            ascending=False,
        )
        .head(10)
        .to_string(index=False)
    )

    print(
        "\nAssignment groups with highest breach volume:"
    )

    print(
        eligible
        .sort_values(
            "sla_breaches",
            ascending=False,
        )[
            [
                "assignment_group",
                "incident_count",
                "sla_breaches",
                "breach_rate",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print(
        "\nAssignment groups with highest median "
        "resolution time:"
    )

    print(
        eligible
        .sort_values(
            "median_resolution_hours",
            ascending=False,
        )[
            [
                "assignment_group",
                "incident_count",
                "median_resolution_hours",
                "p90_resolution_hours",
                "resolution_coverage",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print(
        "\nAssignment-group performance report saved to:"
    )

    print(ASSIGNMENT_GROUP_OUTPUT)


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

def print_assignment_group_bottleneck_summary(
    result: pd.DataFrame,
) -> None:
    """
    Print assignment-group operational bottleneck analysis.
    """

    volume_threshold = result["volume_threshold"].iloc[0]
    breach_rate_threshold = result["breach_rate_threshold"].iloc[0]

    class_counts = (
        result["bottleneck_class"]
        .value_counts()
        .reindex(
            [
                "Critical Bottleneck",
                "Volume Bottleneck",
                "SLA Risk",
                "Lower Priority",
            ],
            fill_value=0,
        )
    )

    critical = result[
        result["bottleneck_class"]
        == "Critical Bottleneck"
    ]

    sla_risk = result[
        result["bottleneck_class"]
        == "SLA Risk"
    ]

    print("\n" + "=" * 70)
    print("ASSIGNMENT-GROUP OPERATIONAL BOTTLENECK ANALYSIS")
    print("=" * 70)

    print(
        f"\nEligible assignment groups: {len(result)}"
    )

    print(
        f"Median incident-volume threshold: "
        f"{volume_threshold:,.0f}"
    )

    print(
        f"Median SLA breach-rate threshold: "
        f"{breach_rate_threshold:.2f}%"
    )

    print("\nBottleneck classification counts:")

    print(
        class_counts.to_string()
    )

    print("\nCritical bottlenecks:")

    if critical.empty:
        print("None")
    else:
        print(
            critical[
                [
                    "assignment_group",
                    "incident_count",
                    "sla_breaches",
                    "breach_rate",
                ]
            ]
            .sort_values(
                by=[
                    "breach_rate",
                    "incident_count",
                ],
                ascending=[
                    False,
                    False,
                ],
            )
            .to_string(index=False)
        )

    print("\nSLA-risk groups:")

    if sla_risk.empty:
        print("None")
    else:
        print(
            sla_risk[
                [
                    "assignment_group",
                    "incident_count",
                    "sla_breaches",
                    "breach_rate",
                ]
            ]
            .sort_values(
                by=[
                    "breach_rate",
                    "incident_count",
                ],
                ascending=[
                    False,
                    False,
                ],
            )
            .to_string(index=False)
        )

    print(
        "\nAssignment-group bottleneck report saved to:"
    )

    print(ASSIGNMENT_GROUP_BOTTLENECK_OUTPUT)


def print_assignment_group_driver_summary(
    result: pd.DataFrame,
) -> None:
    """
    Print assignment-group operational driver analysis.

    Critical Bottleneck groups are compared with all other eligible
    assignment groups using reassignment and resolution-time metrics.
    """

    critical = result[
        result["driver_segment"] == "Critical Bottleneck"
    ]

    other = result[
        result["driver_segment"] == "Other Eligible Groups"
    ]

    print("\n" + "=" * 70)
    print("ASSIGNMENT-GROUP OPERATIONAL DRIVER ANALYSIS")
    print("=" * 70)

    print(
        f"\nCritical Bottleneck groups: {len(critical)}"
    )

    print(
        f"Other eligible groups: {len(other)}"
    )

    if not critical.empty and not other.empty:
        comparison = pd.DataFrame(
            {
                "metric": [
                    "Median breach rate (%)",
                    "Median resolution time (hours)",
                    "Median P90 resolution time (hours)",
                    "Median reassignment rate (%)",
                    "Median mean reassignment count",
                ],
                "critical_bottlenecks": [
                    critical["breach_rate"].median(),
                    critical["median_resolution_hours"].median(),
                    critical["p90_resolution_hours"].median(),
                    critical["reassignment_rate"].median(),
                    critical["mean_reassignment_count"].median(),
                ],
                "other_eligible_groups": [
                    other["breach_rate"].median(),
                    other["median_resolution_hours"].median(),
                    other["p90_resolution_hours"].median(),
                    other["reassignment_rate"].median(),
                    other["mean_reassignment_count"].median(),
                ],
            }
        )

        comparison["difference"] = (
            comparison["critical_bottlenecks"]
            - comparison["other_eligible_groups"]
        )

        print(
            "\nCritical Bottlenecks vs Other Eligible Groups:"
        )

        print(
            comparison.to_string(index=False)
        )

    print(
        "\nAssignment-group driver report saved to:"
    )

    print(ASSIGNMENT_GROUP_DRIVER_OUTPUT)


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
    # Reassignment vs resolution-time analysis
    # -----------------------------------------------------------------------

    reassignment_resolution_result = (
        analyze_reassignment_resolution(
            df
        )
    )

    validate_reassignment_resolution_analysis(
        result=reassignment_resolution_result,
        source_df=df,
    )

    save_reassignment_resolution_report(
        reassignment_resolution_result
    )

    print_reassignment_resolution_summary(
        reassignment_resolution_result
    )

    # -----------------------------------------------------------------------
    # Assignment-group performance analysis
    # -----------------------------------------------------------------------

    assignment_group_result = (
        analyze_assignment_group_performance(
            df
        )
    )

    validate_assignment_group_analysis(
        result=assignment_group_result,
        source_df=df,
    )

    save_assignment_group_report(
        assignment_group_result
    )

    print_assignment_group_summary(
        result=assignment_group_result,
        source_df=df,
    )

    # -----------------------------------------------------------------------
    # Assignment-group operational bottleneck analysis
    # -----------------------------------------------------------------------

    assignment_group_bottleneck_result = (
        analyze_assignment_group_bottlenecks(
            assignment_group_result
        )
    )
    validate_assignment_group_bottlenecks(
        result=assignment_group_bottleneck_result,
        assignment_group_result=assignment_group_result,
    )
    save_assignment_group_bottleneck_report(
        assignment_group_bottleneck_result
    )
    print_assignment_group_bottleneck_summary(
        assignment_group_bottleneck_result
    )

    # -----------------------------------------------------------------------
    # Assignment-group operational driver analysis
    # -----------------------------------------------------------------------

    assignment_group_driver_result = (
        analyze_assignment_group_drivers(
            assignment_group_result=assignment_group_result,
            assignment_group_bottleneck_result=(
                assignment_group_bottleneck_result
            ),
        )
    )

    validate_assignment_group_drivers(
        result=assignment_group_driver_result,
        assignment_group_result=assignment_group_result,
        assignment_group_bottleneck_result=(
            assignment_group_bottleneck_result
        ),
    )

    save_assignment_group_driver_report(
        assignment_group_driver_result
    )

    print_assignment_group_driver_summary(
        assignment_group_driver_result
    )

      # -----------------------------------------------------------------------
      # Incident-level SLA risk reference analysis
      # -----------------------------------------------------------------------

    incident_sla_risk_reference_result = (
        build_incident_sla_risk_reference(
            category_result=category_result,
            priority_result=priority_result,
            reassignment_result=reassignment_result,
            assignment_group_result=assignment_group_result,
        )
    )

    validate_incident_sla_risk_reference(
        result=incident_sla_risk_reference_result,
        category_result=category_result,
        priority_result=priority_result,
        reassignment_result=reassignment_result,
        assignment_group_result=assignment_group_result,
    )

    save_incident_sla_risk_reference_report(
        incident_sla_risk_reference_result
    )



    # -----------------------------------------------------------------------
    # Incident-level SLA risk scoring
    # -----------------------------------------------------------------------

    incident_sla_risk_score_result = (
        build_incident_sla_risk_scores(
            df=df,
            risk_reference=incident_sla_risk_reference_result,
        )
    )

    validate_incident_sla_risk_scores(
        result=incident_sla_risk_score_result,
        source_df=df,
        risk_reference=incident_sla_risk_reference_result,
    )

    save_incident_sla_risk_scores_report(
        incident_sla_risk_score_result
    )

    print(
        "\nIncident-level SLA risk scoring report saved to:"
    )

    print(
        INCIDENT_SLA_RISK_SCORES_OUTPUT
    )

    print(
        f"Scored incidents: "
        f"{len(incident_sla_risk_score_result):,}"
    )

    print(
        f"Mean historical SLA risk score: "
        f"{incident_sla_risk_score_result['sla_risk_score'].mean():.2f}"
    )


    # -----------------------------------------------------------------------
    # Final validation status
    # -----------------------------------------------------------------------

    print("\nCategory validation: PASSED")
    print("Priority validation: PASSED")
    print("Reassignment validation: PASSED")
    print("Reassignment resolution validation: PASSED")
    print("Assignment-group validation: PASSED")
    print("Assignment-group bottleneck validation: PASSED")
    print("Assignment-group driver validation: PASSED")
    print("Incident SLA risk reference validation: PASSED")
    print("Incident SLA risk scoring validation: PASSED")

if __name__ == "__main__":
    main()
