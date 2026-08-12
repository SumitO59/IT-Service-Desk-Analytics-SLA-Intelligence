# IT Service Desk Analytics & SLA Intelligence

> An end-to-end analytics platform for understanding IT service desk performance, SLA compliance, resolution efficiency, reassignment impact, operational bottlenecks, and incident-level SLA risk.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PostgreSQL-18-4169E1?style=for-the-badge&logo=postgresql&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-Analytics-150458?style=for-the-badge&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/SQL-Analytics-336791?style=for-the-badge" />
</p>

---

## Overview

This project transforms **24,918 IT service desk incidents** into an analytical system for monitoring SLA performance, resolution efficiency, reassignment impact, and operational risk.

The workflow combines **Python analytics, PostgreSQL, SQL, SLA risk scoring, analytical reporting, and an interactive Streamlit dashboard**.

```text
Incident Data
     │
     ▼
Data Preparation & Validation
     │
     ▼
PostgreSQL Analytical Database
     │
     ├──────────────► SQL Analysis
     │
     └──────────────► Python Analytics
                          │
                          ▼
                    SLA Risk Scoring
                          │
                          ▼
                   Analytical Reports
                          │
                          ▼
                 Streamlit Dashboard
```

---

## Key Results

| KPI | Result |
|---|---:|
| Total Incidents | **24,918** |
| Resolved Incidents | **23,362** |
| SLA Compliance | **63.42%** |
| SLA Breach Rate | **36.58%** |
| Reassigned Incidents | **11,369** |
| Reassignment Rate | **45.63%** |
| Reopened Incidents | **275** |
| Median Resolution Time | **22.10 hrs** |
| P90 Resolution Time | **381.55 hrs** |

---

## Key Insights

### Priority Risk

Critical and High priority incidents show extremely high SLA exposure.

| Priority | SLA Compliance | SLA Breach |
|---|---:|---:|
| Critical | 1.85% | **98.15%** |
| High | 0.49% | **99.51%** |
| Moderate | 64.54% | 35.46% |
| Low | 84.11% | 15.89% |

### Reassignment Impact

SLA performance deteriorates sharply as reassignment intensity increases.

| Reassignments | SLA Compliance | SLA Breach |
|---|---:|---:|
| 0 | 78.39% | 21.61% |
| 1–2 | 52.46% | 47.54% |
| 3–5 | 28.83% | 71.17% |
| 6+ | **8.55%** | **91.45%** |

This makes reassignment intensity an important operational risk indicator.

### Resolution-Time Distribution

```text
Mean    : 178.17 hrs
Median  : 22.10 hrs
P90     : 381.55 hrs
P95     : 710.52 hrs
P99     : 2843.40 hrs
```

The large difference between the mean and median indicates a strongly right-skewed resolution-time distribution.

---

## Analytical Capabilities

### SLA Analysis

- SLA compliance and breach rates
- Priority-level SLA performance
- Reassignment-level SLA performance
- Category-level SLA exposure
- Assignment-group SLA performance
- Monthly SLA trends
- SLA breach concentration

### Operational Analysis

- Resolution-time distribution
- Median / P90 / P95 / P99 analysis
- Unresolved incident analysis
- Assignment-group bottlenecks
- Reassignment impact
- Monthly workload trends

### SLA Risk Intelligence

The project also performs incident-level risk analysis to identify incidents and operational segments with elevated SLA exposure.

Generated analytical outputs include:

- Incident SLA risk scores
- SLA risk reference data
- SLA risk portfolio summary
- SLA risk concentration analysis
- Assignment-group driver analysis
- Assignment-group bottleneck analysis

---

## PostgreSQL Analytical Layer

PostgreSQL acts as the central analytical database.

The project includes reusable SQL analysis covering:

1. Executive KPI summary
2. SLA performance by priority
3. SLA performance by reassignment
4. Resolution-time distribution
5. Resolution performance by priority
6. SLA performance by category
7. Assignment-group performance
8. Monthly operational performance
9. SLA breach concentration

SQL implementation:

```text
sql/analysis_queries.sql
```

---

## Streamlit Dashboard

The project includes an interactive Streamlit dashboard connected directly to PostgreSQL.

### Dashboard sections

- Executive KPI overview
- SLA performance by priority
- Resolution performance by priority
- SLA performance by reassignment intensity
- Monthly operational performance
- Analytical data tables

Run the dashboard:

```bash
streamlit run app.py
```

---

## Technology Stack

| Layer | Technology |
|---|---|
| Programming | Python |
| Data Analysis | Pandas |
| Database | PostgreSQL |
| Database Driver | Psycopg |
| Database Access | SQLAlchemy |
| Analytics | SQL |
| Dashboard | Streamlit |
| Exploration | Jupyter Notebook |
| Version Control | Git / GitHub |

---

## Project Structure

```text
IT-Service-Desk-Analytics/
│
├── app.py
├── README.md
├── requirements.txt
│
├── src/
│   ├── database.py
│   ├── data_cleaning.py
│   ├── metrics.py
│   └── sla_analysis.py
│
├── sql/
│   └── analysis_queries.sql
│
├── reports/
│   ├── incident_sla_risk_scores.csv
│   ├── incident_sla_risk_reference.csv
│   ├── sla_risk_portfolio_summary.csv
│   ├── sla_risk_concentration_analysis.csv
│   ├── assignment_group_performance.csv
│   ├── assignment_group_bottleneck_analysis.csv
│   ├── assignment_group_driver_analysis.csv
│   ├── category_sla_performance.csv
│   ├── priority_sla_performance.csv
│   ├── reassignment_sla_performance.csv
│   ├── reassignment_resolution_performance.csv
│   └── monthly_operational_performance.csv
│
└── notebooks/
    └── service_desk_analysis.ipynb
```

---

## Jupyter Notebook

The Jupyter notebook was used during the **exploratory analysis and validation phase**.

It was used for:

- Dataset inspection
- Data-quality investigation
- Exploratory analysis
- Distribution analysis
- Relationship exploration
- Prototyping analytical logic
- Validating metrics before production implementation

Validated logic was subsequently moved into reusable Python modules, SQL queries, reports, and the Streamlit application.

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/SumitO59/IT-Service-Desk-Analytics-SLA-Intelligence.git
cd IT-Service-Desk-Analytics-SLA-Intelligence
```

### 2. Create / activate the environment

```bash
conda activate service-desk
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure PostgreSQL

The application expects:

```text
Host: localhost
Port: 5432
Database: service_desk_analytics
User: service_desk_app
```

### 5. Run the dashboard

```bash
streamlit run app.py
```

---

## Validation

Compile the Streamlit application:

```bash
python -m py_compile app.py
```

Validate SQL:

```bash
PAGER=cat psql \
  -h localhost \
  -p 5432 \
  -U service_desk_app \
  -d service_desk_analytics \
  -f sql/analysis_queries.sql
```

Check Git formatting:

```bash
git diff --check
```


---

## Author

**Sumit Salgotra**

B.Tech Computer Science & Engineering
National Institute of Technology Srinagar

---

## License

This project is intended for educational, analytical, and portfolio purposes.