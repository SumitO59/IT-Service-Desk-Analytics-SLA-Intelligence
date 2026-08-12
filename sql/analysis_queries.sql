-- ============================================================================
-- IT SERVICE DESK ANALYTICS — SQL ANALYSIS
-- PostgreSQL 18
-- ============================================================================


-- ============================================================================
-- Q01: Executive KPI Summary
-- ============================================================================
--
-- Business question:
-- What is the overall operational state of the service desk?
--
-- Analytical purpose:
-- Establish the core workload, SLA, resolution, reassignment, and
-- reopening KPIs directly from the PostgreSQL analytical dataset.
--
-- SQL techniques:
-- COUNT, FILTER, conditional aggregation, ROUND, derived metrics.
-- ============================================================================

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


-- ============================================================================
-- Q02: SLA Performance by Priority
-- ============================================================================
--
-- Business question:
-- How does SLA performance vary across incident priorities?
--
-- Analytical purpose:
-- Compare incident volume, SLA compliance, and SLA breach rates across
-- operational priority levels.
--
-- SQL techniques:
-- CASE, GROUP BY, COUNT, FILTER, ROUND, conditional aggregation.
-- ============================================================================

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

-- ============================================================================
-- Q03: SLA Performance by Reassignment Bucket
-- ============================================================================
--
-- Business question:
-- How does reassignment intensity affect SLA performance?
--
-- Analytical purpose:
-- Quantify the relationship between incident reassignment volume and
-- SLA breach exposure.
--
-- SQL techniques:
-- CTE, CASE, GROUP BY, FILTER, ROUND, conditional aggregation.
-- ============================================================================

WITH reassignment_classified AS (
    SELECT
        CASE
            WHEN reassignment_bucket = '0' THEN '0'
            WHEN reassignment_bucket = '1-2' THEN '1-2'
            WHEN reassignment_bucket = '3-5' THEN '3-5'
            WHEN reassignment_bucket = '6+' THEN '6+'
            ELSE 'Unknown'
        END AS reassignment_level,
        made_sla,
        sla_breached
    FROM incidents
)

SELECT
    reassignment_level,

    COUNT(*) AS incident_count,

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
    ) AS sla_breach_rate_pct

FROM reassignment_classified

GROUP BY reassignment_level

ORDER BY
    CASE reassignment_level
        WHEN '0' THEN 1
        WHEN '1-2' THEN 2
        WHEN '3-5' THEN 3
        WHEN '6+' THEN 4
        ELSE 5
    END;

    -- ============================================================================
-- Q04: Resolution-Time Distribution
-- ============================================================================
--
-- Business question:
-- What does the distribution of incident resolution time look like?
--
-- Analytical purpose:
-- Measure central tendency and long-tail resolution behavior using
-- PostgreSQL's continuous percentile functions.
--
-- SQL techniques:
-- Aggregate functions, percentile_cont, ROUND, NULL filtering.
-- ============================================================================

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

-- ============================================================================
-- Q05: Resolution Performance by Priority
-- ============================================================================
--
-- Business question:
-- How does resolution time vary across incident priorities?
--
-- Analytical purpose:
-- Compare incident volume, median resolution time, P90 resolution time,
-- and unresolved workload across operational priority levels.
--
-- SQL techniques:
-- CTE, CASE, GROUP BY, FILTER, percentile_cont, ROUND.
-- ============================================================================

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

    -- ============================================================================
-- Q06: SLA Performance by Category
-- ============================================================================
--
-- Business question:
-- Which incident categories have the highest SLA breach exposure?
--
-- Analytical purpose:
-- Identify high-volume categories with poor SLA performance.
--
-- SQL techniques:
-- GROUP BY, FILTER, COUNT, ROUND, conditional aggregation.
-- ============================================================================

SELECT
    category,

    COUNT(*) AS incident_count,

    COUNT(*) FILTER (
        WHERE sla_breached = TRUE
    ) AS sla_breached_incidents,

    ROUND(
        100.0 * COUNT(*) FILTER (
            WHERE sla_breached = TRUE
        ) / COUNT(*),
        2
    ) AS sla_breach_rate_pct,

    ROUND(
        AVG(resolution_time_hours)::numeric,
        2
    ) AS mean_resolution_hours

FROM incidents

GROUP BY category

HAVING COUNT(*) >= 100

ORDER BY
    sla_breach_rate_pct DESC,
    incident_count DESC;

    -- ============================================================================
-- Q07: Assignment Group Performance
-- ============================================================================
--
-- Business question:
-- Which assignment groups have the greatest SLA breach exposure?
--
-- Analytical purpose:
-- Identify high-volume operational teams with poor SLA performance.
--
-- SQL techniques:
-- GROUP BY, FILTER, COUNT, ROUND, conditional aggregation.
-- ============================================================================

SELECT
    COALESCE(assignment_group, 'Unknown') AS assignment_group,

    COUNT(*) AS incident_count,

    COUNT(*) FILTER (
        WHERE sla_breached = TRUE
    ) AS sla_breached_incidents,

    ROUND(
        100.0 * COUNT(*) FILTER (
            WHERE sla_breached = TRUE
        ) / COUNT(*),
        2
    ) AS sla_breach_rate_pct,

    ROUND(
        AVG(resolution_time_hours)::numeric,
        2
    ) AS mean_resolution_hours

FROM incidents

GROUP BY COALESCE(assignment_group, 'Unknown')

HAVING COUNT(*) >= 100

ORDER BY
    sla_breach_rate_pct DESC,
    incident_count DESC;

    -- ============================================================================
-- Q08: Monthly Operational Performance
-- ============================================================================
--
-- Business question:
-- How does service desk workload and SLA performance change over time?
--
-- Analytical purpose:
-- Identify monthly changes in incident volume, SLA breaches,
-- SLA compliance, and resolution performance.
--
-- SQL techniques:
-- GROUP BY, DATE_TRUNC, FILTER, COUNT, ROUND, AVG.
-- ============================================================================

SELECT
    opened_month,

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

-- ============================================================================
-- Q09: SLA Breach Concentration by Priority and Reassignment
-- ============================================================================
--
-- Business question:
-- Where is SLA breach exposure concentrated across priority and
-- reassignment intensity?
--
-- Analytical purpose:
-- Identify operational combinations associated with high SLA risk.
--
-- SQL techniques:
-- GROUP BY, FILTER, COUNT, ROUND, conditional aggregation.
-- ============================================================================

SELECT
    priority,
    reassignment_bucket,

    COUNT(*) AS incident_count,

    COUNT(*) FILTER (
        WHERE sla_breached = TRUE
    ) AS sla_breached_incidents,

    ROUND(
        100.0 * COUNT(*) FILTER (
            WHERE sla_breached = TRUE
        ) / COUNT(*),
        2
    ) AS sla_breach_rate_pct,

    ROUND(
        100.0 * COUNT(*) FILTER (
            WHERE sla_breached = TRUE
        ) / SUM(
            COUNT(*) FILTER (
                WHERE sla_breached = TRUE
            )
        ) OVER (),
        2
    ) AS share_of_all_breaches_pct

FROM incidents

GROUP BY
    priority,
    reassignment_bucket

HAVING COUNT(*) >= 50

ORDER BY
    sla_breached_incidents DESC,
    sla_breach_rate_pct DESC;