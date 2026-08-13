import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="IT Service Desk Analytics",
    page_icon="📊",
    layout="wide",
)


# ============================================================================
# DATABASE CONFIGURATION
# ============================================================================

DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "service_desk_analytics"
DB_USER = "service_desk_app"


@st.cache_resource
def get_engine():
    """Create and cache the PostgreSQL SQLAlchemy engine."""

    connection_url = (
        f"postgresql+psycopg://"
        f"{DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )

    return create_engine(connection_url)


@st.cache_data
def load_kpi_data():
    """Load executive KPI metrics from PostgreSQL."""

    query = """
        SELECT
            COUNT(*) AS total_incidents,

            COUNT(*) FILTER (
                WHERE resolved_at IS NOT NULL
            ) AS resolved_incidents,

            COUNT(*) FILTER (
                WHERE made_sla = TRUE
            ) AS sla_met_incidents,

            COUNT(*) FILTER (
                WHERE sla_breached = TRUE
            ) AS sla_breached_incidents,

            ROUND(
                100.0 * COUNT(*) FILTER (
                    WHERE made_sla = TRUE
                ) / COUNT(*),
                2
            ) AS sla_compliance_rate_pct,

            ROUND(
                100.0 * COUNT(*) FILTER (
                    WHERE sla_breached = TRUE
                ) / COUNT(*),
                2
            ) AS sla_breach_rate_pct,

            COUNT(*) FILTER (
                WHERE reassignment_count > 0
            ) AS reassigned_incidents,

            ROUND(
                100.0 * COUNT(*) FILTER (
                    WHERE reassignment_count > 0
                ) / COUNT(*),
                2
            ) AS reassignment_rate_pct,

            COUNT(*) FILTER (
                WHERE reopened = TRUE
            ) AS reopened_incidents,

            ROUND(
                100.0 * COUNT(*) FILTER (
                    WHERE reopened = TRUE
                ) / COUNT(*),
                2
            ) AS reopen_rate_pct

        FROM incidents;
    """

    engine = get_engine()

    with engine.connect() as connection:
        return pd.read_sql(text(query), connection)


# ============================================================================
# DASHBOARD HEADER
# ============================================================================

st.title("IT Service Desk Analytics")
st.caption(
    "PostgreSQL-backed operational intelligence for SLA, workload, "
    "resolution, and service desk performance."
)


# ============================================================================
# DATABASE CONNECTION TEST
# ============================================================================

try:
    kpi_data = load_kpi_data()
except Exception as exc:
    st.error("Unable to connect to the PostgreSQL analytical database.")
    st.exception(exc)
    st.stop()


# ============================================================================
# EXECUTIVE OVERVIEW
# ============================================================================

st.subheader("Executive Overview")

kpi = kpi_data.iloc[0]

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Incidents",
        f"{int(kpi['total_incidents']):,}",
    )

with col2:
    st.metric(
        "Resolved Incidents",
        f"{int(kpi['resolved_incidents']):,}",
    )

with col3:
    st.metric(
        "SLA Compliance",
        f"{kpi['sla_compliance_rate_pct']:.2f}%",
    )

with col4:
    st.metric(
        "SLA Breach Rate",
        f"{kpi['sla_breach_rate_pct']:.2f}%",
    )


# ============================================================================
# SECONDARY KPIs
# ============================================================================

st.subheader("Operational Indicators")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "SLA Breaches",
        f"{int(kpi['sla_breached_incidents']):,}",
    )

with col2:
    st.metric(
        "Reassigned Incidents",
        f"{int(kpi['reassigned_incidents']):,}",
    )

with col3:
    st.metric(
        "Reassignment Rate",
        f"{kpi['reassignment_rate_pct']:.2f}%",
    )

with col4:
    st.metric(
        "Reopened Incidents",
        f"{int(kpi['reopened_incidents']):,}",
    )


# ============================================================================
# DATA VALIDATION
# ============================================================================

st.caption(
    f"PostgreSQL connection active • "
    f"{int(kpi['total_incidents']):,} incidents available for analysis."
)

# ============================================================================
# SLA PERFORMANCE BY PRIORITY
# ============================================================================

@st.cache_data
def load_priority_sla_data():
    """Load SLA performance metrics grouped by incident priority."""

    query = """
        WITH priority_classified AS (
            SELECT
                CASE
                    WHEN priority = '1 - Critical' THEN 'Critical'
                    WHEN priority = '2 - High' THEN 'High'
                    WHEN priority = '3 - Moderate' THEN 'Moderate'
                    WHEN priority = '4 - Low' THEN 'Low'
                    ELSE 'Unknown'
                END AS priority_level,
                made_sla,
                sla_breached
            FROM incidents
        )

        SELECT
            priority_level,

            COUNT(*) AS incident_count,

            ROUND(
                100.0 * COUNT(*) FILTER (
                    WHERE made_sla = TRUE
                ) / COUNT(*),
                2
            ) AS sla_compliance_rate_pct,

            ROUND(
                100.0 * COUNT(*) FILTER (
                    WHERE sla_breached = TRUE
                ) / COUNT(*),
                2
            ) AS sla_breach_rate_pct

        FROM priority_classified

        GROUP BY priority_level

        ORDER BY
            CASE priority_level
                WHEN 'Critical' THEN 1
                WHEN 'High' THEN 2
                WHEN 'Moderate' THEN 3
                WHEN 'Low' THEN 4
                ELSE 5
            END;
    """

    engine = get_engine()

    with engine.connect() as connection:
        return pd.read_sql(text(query), connection)


priority_sla = load_priority_sla_data()

st.divider()

st.subheader("SLA Performance by Priority")

st.caption(
    "SLA compliance and breach exposure across operational priority levels."
)

col1, col2 = st.columns(2)

with col1:
    st.markdown("**SLA Compliance Rate**")

    compliance_chart = priority_sla.set_index(
        "priority_level"
    )["sla_compliance_rate_pct"]

    st.bar_chart(
        compliance_chart,
        y_label="SLA Compliance (%)",
        x_label="Priority",
    )

with col2:
    st.markdown("**SLA Breach Rate**")

    breach_chart = priority_sla.set_index(
        "priority_level"
    )["sla_breach_rate_pct"]

    st.bar_chart(
        breach_chart,
        y_label="SLA Breach (%)",
        x_label="Priority",
    )

st.dataframe(
    priority_sla,
    width="stretch",
    hide_index=True,
)

highest_breach_priority = priority_sla.loc[
    priority_sla["sla_breach_rate_pct"].idxmax()
]

st.info(
    f"Highest SLA breach exposure: "
    f"{highest_breach_priority['priority_level']} priority incidents "
    f"with a {highest_breach_priority['sla_breach_rate_pct']:.2f}% "
    f"breach rate."
)

# ============================================================================
# SLA PERFORMANCE BY REASSIGNMENT INTENSITY
# ============================================================================

@st.cache_data
def load_reassignment_sla_data():
    """Load SLA performance grouped by reassignment intensity."""

    query = """
        SELECT
            reassignment_bucket AS reassignment_level,

            COUNT(*) AS incident_count,

            COUNT(*) FILTER (
                WHERE sla_breached = TRUE
            ) AS sla_breached_incidents,

            ROUND(
                100.0 * COUNT(*) FILTER (
                    WHERE made_sla = TRUE
                ) / COUNT(*),
                2
            ) AS sla_compliance_rate_pct,

            ROUND(
                100.0 * COUNT(*) FILTER (
                    WHERE sla_breached = TRUE
                ) / COUNT(*),
                2
            ) AS sla_breach_rate_pct

        FROM incidents

        GROUP BY reassignment_bucket

        ORDER BY
            CASE reassignment_bucket
                WHEN '0' THEN 1
                WHEN '1-2' THEN 2
                WHEN '3-5' THEN 3
                WHEN '6+' THEN 4
                ELSE 5
            END;
    """

    engine = get_engine()

    with engine.connect() as connection:
        return pd.read_sql(text(query), connection)


reassignment_sla = load_reassignment_sla_data()

st.divider()

st.subheader("SLA Performance by Reassignment Intensity")

st.caption(
    "SLA performance deteriorates sharply as incidents move through "
    "multiple reassignment cycles."
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**SLA Compliance Rate**")

    compliance_chart = reassignment_sla.set_index(
        "reassignment_level"
    )["sla_compliance_rate_pct"]

    st.bar_chart(
        compliance_chart,
        y_label="SLA Compliance (%)",
        x_label="Reassignment Bucket",
    )

with col2:
    st.markdown("**SLA Breach Rate**")

    breach_chart = reassignment_sla.set_index(
        "reassignment_level"
    )["sla_breach_rate_pct"]

    st.bar_chart(
        breach_chart,
        y_label="SLA Breach (%)",
        x_label="Reassignment Bucket",
    )

with col3:
    st.markdown("**Incident Volume**")

    reassignment_volume_chart = reassignment_sla.set_index(
        "reassignment_level"
    )["incident_count"]

    st.bar_chart(
        reassignment_volume_chart,
        y_label="Incidents",
        x_label="Reassignment Bucket",
    )

st.dataframe(
    reassignment_sla,
    width="stretch",
    hide_index=True,
)

highest_reassignment_risk = reassignment_sla.loc[
    reassignment_sla["sla_breach_rate_pct"].idxmax()
]

st.warning(
    f"Highest reassignment risk: incidents in the "
    f"{highest_reassignment_risk['reassignment_level']} reassignment bucket "
    f"have a {highest_reassignment_risk['sla_breach_rate_pct']:.2f}% "
    f"SLA breach rate."
)


# ============================================================================
# RESOLUTION-TIME PERFORMANCE
# ============================================================================

@st.cache_data
def load_resolution_data():
    """Load overall resolution-time distribution metrics."""

    query = """
        SELECT
            COUNT(*) AS resolved_incidents,

            ROUND(
                AVG(resolution_time_hours)::numeric,
                2
            ) AS mean_resolution_hours,

            ROUND(
                percentile_cont(0.50)
                WITHIN GROUP (
                    ORDER BY resolution_time_hours
                )::numeric,
                2
            ) AS median_resolution_hours,

            ROUND(
                percentile_cont(0.90)
                WITHIN GROUP (
                    ORDER BY resolution_time_hours
                )::numeric,
                2
            ) AS p90_resolution_hours,

            ROUND(
                percentile_cont(0.95)
                WITHIN GROUP (
                    ORDER BY resolution_time_hours
                )::numeric,
                2
            ) AS p95_resolution_hours,

            ROUND(
                percentile_cont(0.99)
                WITHIN GROUP (
                    ORDER BY resolution_time_hours
                )::numeric,
                2
            ) AS p99_resolution_hours

        FROM incidents

        WHERE resolution_time_hours IS NOT NULL;
    """

    engine = get_engine()

    with engine.connect() as connection:
        return pd.read_sql(text(query), connection)


resolution_data = load_resolution_data().iloc[0]

st.divider()

st.subheader("Resolution-Time Performance")

st.caption(
    "Resolution-time distribution highlights the typical incident experience "
    "and the long tail of complex cases."
)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Median",
        f"{resolution_data['median_resolution_hours']:.2f} h",
    )

with col2:
    st.metric(
        "Mean",
        f"{resolution_data['mean_resolution_hours']:.2f} h",
    )

with col3:
    st.metric(
        "P90",
        f"{resolution_data['p90_resolution_hours']:.2f} h",
    )

with col4:
    st.metric(
        "P95",
        f"{resolution_data['p95_resolution_hours']:.2f} h",
    )

with col5:
    st.metric(
        "P99",
        f"{resolution_data['p99_resolution_hours']:.2f} h",
    )


# ============================================================================
# RESOLUTION PERFORMANCE BY PRIORITY
# ============================================================================

@st.cache_data
def load_priority_resolution_data():
    """Load resolution performance grouped by priority."""

    query = """
        WITH priority_classified AS (
            SELECT
                CASE
                    WHEN priority = '1 - Critical' THEN 'Critical'
                    WHEN priority = '2 - High' THEN 'High'
                    WHEN priority = '3 - Moderate' THEN 'Moderate'
                    WHEN priority = '4 - Low' THEN 'Low'
                    ELSE 'Unknown'
                END AS priority_level,
                resolution_time_hours
            FROM incidents
        )

        SELECT
            priority_level,

            COUNT(*) AS incident_count,

            COUNT(*) FILTER (
                WHERE resolution_time_hours IS NOT NULL
            ) AS resolved_incidents,

            COUNT(*) FILTER (
                WHERE resolution_time_hours IS NULL
            ) AS unresolved_incidents,

            ROUND(
                100.0 * COUNT(*) FILTER (
                    WHERE resolution_time_hours IS NULL
                ) / COUNT(*),
                2
            ) AS unresolved_rate_pct,

            ROUND(
                percentile_cont(0.50)
                WITHIN GROUP (
                    ORDER BY resolution_time_hours
                )::numeric,
                2
            ) AS median_resolution_hours,

            ROUND(
                percentile_cont(0.90)
                WITHIN GROUP (
                    ORDER BY resolution_time_hours
                )::numeric,
                2
            ) AS p90_resolution_hours

        FROM priority_classified

        GROUP BY priority_level

        ORDER BY
            CASE priority_level
                WHEN 'Critical' THEN 1
                WHEN 'High' THEN 2
                WHEN 'Moderate' THEN 3
                WHEN 'Low' THEN 4
                ELSE 5
            END;
    """

    engine = get_engine()

    with engine.connect() as connection:
        return pd.read_sql(text(query), connection)


priority_resolution = load_priority_resolution_data()

st.markdown("### Resolution Performance by Priority")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Median Resolution Time**")

    median_resolution_chart = priority_resolution.set_index(
        "priority_level"
    )["median_resolution_hours"]

    st.bar_chart(
        median_resolution_chart,
        y_label="Hours",
        x_label="Priority",
    )

with col2:
    st.markdown("**P90 Resolution Time**")

    p90_resolution_chart = priority_resolution.set_index(
        "priority_level"
    )["p90_resolution_hours"]

    st.bar_chart(
        p90_resolution_chart,
        y_label="Hours",
        x_label="Priority",
    )

st.dataframe(
    priority_resolution,
    width="stretch",
    hide_index=True,
)

# ============================================================================
# MONTHLY OPERATIONAL PERFORMANCE
# ============================================================================

st.divider()

st.subheader("Monthly Operational Performance")

st.caption(
    "Monthly workload, SLA performance, and resolution trends across the "
    "analytical dataset."
)


@st.cache_data
def load_monthly_operational_data():
    """Load monthly service desk performance metrics."""

    query = """
        SELECT
            opened_month,

            COUNT(*) AS incident_count,

            ROUND(
                100.0 * COUNT(*) FILTER (
                    WHERE made_sla = TRUE
                ) / COUNT(*),
                2
            ) AS sla_compliance_rate_pct,

            ROUND(
                100.0 * COUNT(*) FILTER (
                    WHERE sla_breached = TRUE
                ) / COUNT(*),
                2
            ) AS sla_breach_rate_pct,

            COUNT(*) FILTER (
                WHERE resolution_time_hours IS NOT NULL
            ) AS resolved_incidents,

            ROUND(
                AVG(resolution_time_hours)::numeric,
                2
            ) AS mean_resolution_hours

        FROM incidents

        GROUP BY opened_month

        ORDER BY opened_month;
    """

    engine = get_engine()

    with engine.connect() as connection:
        return pd.read_sql(text(query), connection)


monthly_operational = load_monthly_operational_data()

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Monthly Incident Volume**")

    incident_volume_chart = monthly_operational.set_index(
        "opened_month"
    )["incident_count"]

    st.line_chart(
        incident_volume_chart,
        y_label="Incidents",
        x_label="Month",
    )

with col2:
    st.markdown("**Monthly SLA Compliance**")

    monthly_sla_chart = monthly_operational.set_index(
        "opened_month"
    )["sla_compliance_rate_pct"]

    st.line_chart(
        monthly_sla_chart,
        y_label="SLA Compliance (%)",
        x_label="Month",
    )

st.markdown("**Monthly Resolution Performance**")

monthly_resolution_chart = monthly_operational.set_index(
    "opened_month"
)[["mean_resolution_hours"]]

st.line_chart(
    monthly_resolution_chart,
    y_label="Mean Resolution Hours",
    x_label="Month",
)

st.dataframe(
    monthly_operational,
    width="stretch",
    hide_index=True,
)
