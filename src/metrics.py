from pathlib import Path

import pandas as pd

from src.database import get_connection


# -------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------

def load_incidents() -> pd.DataFrame:
    """
    Load the cleaned incident-level dataset from PostgreSQL.

    Returns
    -------
    pandas.DataFrame
        One row per incident.
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT *
                FROM incidents
                ORDER BY number
            """)

            rows = cur.fetchall()
            columns = [description.name for description in cur.description]

    df = pd.DataFrame(rows, columns=columns)

    timestamp_columns = [
        "opened_at",
        "resolved_at",
        "closed_at",
    ]

    for column in timestamp_columns:
        df[column] = pd.to_datetime(
            df[column],
            errors="coerce",
        )

    return df


# -------------------------------------------------------------------
# Overall KPI metrics
# -------------------------------------------------------------------

def calculate_overall_kpis(
    df: pd.DataFrame,
) -> dict:
    """
    Calculate executive-level service desk KPIs.
    """

    total_incidents = len(df)

    sla_met = int(
        df["made_sla"]
        .eq(True)
        .sum()
    )

    sla_breached = int(
        df["sla_breached"]
        .eq(True)
        .sum()
    )

    resolved_incidents = int(
        df["resolved_at"]
        .notna()
        .sum()
    )

    missing_resolution_timestamp = (
        total_incidents
        - resolved_incidents
    )

    reassigned_incidents = int(
        df["reassignment_count"]
        .gt(0)
        .sum()
    )

    reopened_incidents = int(
        df["reopened"]
        .eq(True)
        .sum()
    )

    return {
        "total_incidents": total_incidents,

        "sla_met": sla_met,
        "sla_breached": sla_breached,

        "sla_compliance_rate": (
            sla_met / total_incidents * 100
            if total_incidents > 0
            else 0.0
        ),

        "sla_breach_rate": (
            sla_breached / total_incidents * 100
            if total_incidents > 0
            else 0.0
        ),

        "resolved_incidents": resolved_incidents,

        "missing_resolution_timestamp": (
            missing_resolution_timestamp
        ),

        "resolution_coverage_rate": (
            resolved_incidents / total_incidents * 100
            if total_incidents > 0
            else 0.0
        ),

        "reassigned_incidents": reassigned_incidents,

        "reassignment_rate": (
            reassigned_incidents / total_incidents * 100
            if total_incidents > 0
            else 0.0
        ),

        "reopened_incidents": reopened_incidents,

        "reopen_rate": (
            reopened_incidents / total_incidents * 100
            if total_incidents > 0
            else 0.0
        ),
    }


# -------------------------------------------------------------------
# Resolution metrics
# -------------------------------------------------------------------

def calculate_resolution_metrics(
    df: pd.DataFrame,
) -> dict:
    """
    Calculate resolution-time statistics.

    Incidents without a resolved_at timestamp are excluded
    from resolution-time calculations.
    """

    resolved = df[
        df["resolution_time_hours"].notna()
    ]["resolution_time_hours"]

    if resolved.empty:
        return {
            "mean_resolution_hours": None,
            "median_resolution_hours": None,
            "p90_resolution_hours": None,
            "p95_resolution_hours": None,
            "p99_resolution_hours": None,
        }

    return {
        "mean_resolution_hours": resolved.mean(),

        "median_resolution_hours": (
            resolved.median()
        ),

        "p90_resolution_hours": (
            resolved.quantile(0.90)
        ),

        "p95_resolution_hours": (
            resolved.quantile(0.95)
        ),

        "p99_resolution_hours": (
            resolved.quantile(0.99)
        ),
    }


# -------------------------------------------------------------------
# Closure metrics
# -------------------------------------------------------------------

def calculate_closure_metrics(
    df: pd.DataFrame,
) -> dict:
    """
    Calculate closure-time and closure-lag statistics.

    Closure lag is the time between technical resolution
    and formal incident closure.
    """

    closure = df[
        df["closure_time_hours"].notna()
    ]["closure_time_hours"]

    closure_lag = df[
        df["resolved_at"].notna()
        & df["closed_at"].notna()
    ].copy()

    if not closure.empty:

        closure_stats = {
            "mean_closure_hours": (
                closure.mean()
            ),

            "median_closure_hours": (
                closure.median()
            ),

            "p90_closure_hours": (
                closure.quantile(0.90)
            ),

            "p95_closure_hours": (
                closure.quantile(0.95)
            ),
        }

    else:

        closure_stats = {
            "mean_closure_hours": None,
            "median_closure_hours": None,
            "p90_closure_hours": None,
            "p95_closure_hours": None,
        }

    if not closure_lag.empty:

        closure_lag_hours = (
            closure_lag["closed_at"]
            - closure_lag["resolved_at"]
        ).dt.total_seconds() / 3600

        closure_stats.update({
            "median_closure_lag_hours": (
                closure_lag_hours.median()
            ),

            "p90_closure_lag_hours": (
                closure_lag_hours.quantile(0.90)
            ),
        })

    else:

        closure_stats.update({
            "median_closure_lag_hours": None,
            "p90_closure_lag_hours": None,
        })

    return closure_stats


# -------------------------------------------------------------------
# Reassignment metrics
# -------------------------------------------------------------------

def calculate_reassignment_metrics(
    df: pd.DataFrame,
) -> dict:
    """
    Calculate reassignment statistics.
    """

    reassignment = df[
        "reassignment_count"
    ]

    return {
        "mean_reassignments": (
            reassignment.mean()
        ),

        "median_reassignments": (
            reassignment.median()
        ),

        "p90_reassignments": (
            reassignment.quantile(0.90)
        ),

        "p95_reassignments": (
            reassignment.quantile(0.95)
        ),

        "max_reassignments": (
            reassignment.max()
        ),
    }


# -------------------------------------------------------------------
# Reopen metrics
# -------------------------------------------------------------------

def calculate_reopen_metrics(
    df: pd.DataFrame,
) -> dict:
    """
    Calculate incident reopening statistics.
    """

    reopen = df["reopen_count"]

    return {
        "mean_reopens": reopen.mean(),

        "median_reopens": (
            reopen.median()
        ),

        "p90_reopens": (
            reopen.quantile(0.90)
        ),

        "max_reopens": (
            reopen.max()
        ),
    }


# -------------------------------------------------------------------
# Monthly metrics
# -------------------------------------------------------------------

def calculate_monthly_metrics(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate monthly incident volume,
    SLA performance, and resolution metrics.
    """

    monthly = (
        df.groupby("opened_month")
        .agg(
            incident_count=(
                "number",
                "count",
            ),

            sla_breached=(
                "sla_breached",
                "sum",
            ),

            sla_met=(
                "made_sla",
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
        )
        .reset_index()
    )

    monthly["sla_compliance_rate"] = (
        monthly["sla_met"]
        / monthly["incident_count"]
        * 100
    )

    monthly["sla_breach_rate"] = (
        monthly["sla_breached"]
        / monthly["incident_count"]
        * 100
    )

    return monthly


# -------------------------------------------------------------------
# Category metrics
# -------------------------------------------------------------------

def calculate_category_metrics(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate service-desk performance metrics
    by incident category.
    """

    category = (
        df.groupby(
            "category",
            dropna=False,
        )
        .agg(
            incident_count=(
                "number",
                "count",
            ),

            sla_breached=(
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

            reassignment_rate=(
                "reassignment_count",
                lambda x: (
                    (x > 0).mean() * 100
                ),
            ),

            reopen_rate=(
                "reopened",
                "mean",
            ),
        )
        .reset_index()
    )

    category["sla_breach_rate"] = (
        category["sla_breached"]
        / category["incident_count"]
        * 100
    )

    category["reopen_rate"] = (
        category["reopen_rate"] * 100
    )

    return category.sort_values(
        "incident_count",
        ascending=False,
    )


# -------------------------------------------------------------------
# Assignment-group metrics
# -------------------------------------------------------------------

def calculate_assignment_group_metrics(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate service-desk performance metrics
    by assignment group.
    """

    assignment = (
        df.groupby(
            "assignment_group",
            dropna=False,
        )
        .agg(
            incident_count=(
                "number",
                "count",
            ),

            sla_breached=(
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

            reassignment_rate=(
                "reassignment_count",
                lambda x: (
                    (x > 0).mean() * 100
                ),
            ),
        )
        .reset_index()
    )

    assignment["sla_breach_rate"] = (
        assignment["sla_breached"]
        / assignment["incident_count"]
        * 100
    )

    return assignment.sort_values(
        "incident_count",
        ascending=False,
    )


# -------------------------------------------------------------------
# Main diagnostic execution
# -------------------------------------------------------------------

if __name__ == "__main__":

    incidents = load_incidents()

    print("=" * 70)
    print("IT SERVICE DESK — ANALYTICAL METRICS")
    print("=" * 70)

    print(
        f"\nLoaded incidents: "
        f"{len(incidents):,}"
    )

    # ---------------------------------------------------------------
    # Overall KPIs
    # ---------------------------------------------------------------

    print("\nOverall KPIs:")

    overall = calculate_overall_kpis(
        incidents
    )

    for metric, value in overall.items():

        if isinstance(value, float):

            print(
                f"{metric}: "
                f"{value:.2f}"
            )

        else:

            print(
                f"{metric}: "
                f"{value:,}"
            )

    # ---------------------------------------------------------------
    # Resolution metrics
    # ---------------------------------------------------------------

    print("\nResolution metrics:")

    resolution = calculate_resolution_metrics(
        incidents
    )

    for metric, value in resolution.items():

        if value is None:

            print(
                f"{metric}: N/A"
            )

        else:

            print(
                f"{metric}: "
                f"{value:.2f} hours"
            )

    # ---------------------------------------------------------------
    # Closure metrics
    # ---------------------------------------------------------------

    print("\nClosure metrics:")

    closure = calculate_closure_metrics(
        incidents
    )

    for metric, value in closure.items():

        if value is None:

            print(
                f"{metric}: N/A"
            )

        else:

            print(
                f"{metric}: "
                f"{value:.2f} hours"
            )

    # ---------------------------------------------------------------
    # Reassignment metrics
    # ---------------------------------------------------------------

    print("\nReassignment metrics:")

    reassignment = (
        calculate_reassignment_metrics(
            incidents
        )
    )

    for metric, value in reassignment.items():

        print(
            f"{metric}: "
            f"{value:.2f}"
        )

    # ---------------------------------------------------------------
    # Reopen metrics
    # ---------------------------------------------------------------

    print("\nReopen metrics:")

    reopen = calculate_reopen_metrics(
        incidents
    )

    for metric, value in reopen.items():

        print(
            f"{metric}: "
            f"{value:.4f}"
        )

    # ---------------------------------------------------------------
    # Category metrics
    # ---------------------------------------------------------------

    print("\nTop categories:")

    category = calculate_category_metrics(
        incidents
    )

    print(
        category
        .head(10)
        .to_string(index=False)
    )

    # ---------------------------------------------------------------
    # Assignment-group metrics
    # ---------------------------------------------------------------

    print("\nTop assignment groups:")

    assignment = (
        calculate_assignment_group_metrics(
            incidents
        )
    )

    print(
        assignment
        .head(10)
        .to_string(index=False)
    )

    print(
        "\nMetrics calculation complete."
    )
