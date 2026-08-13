from pathlib import Path
import os

import psycopg


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "incidents_clean.csv"
)

SCHEMA_FILE = (
    PROJECT_ROOT
    / "sql"
    / "schema"
    / "create_incidents_table.sql"
)


# ---------------------------------------------------------------------------
# PostgreSQL configuration
# ---------------------------------------------------------------------------

DATABASE_URL = os.environ.get("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL environment variable is not configured."
    )

# ---------------------------------------------------------------------------
# Expected analytical schema
# ---------------------------------------------------------------------------

EXPECTED_COLUMNS = [
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
    "opened_by",
    "sys_created_by",
    "sys_created_at",
    "sys_updated_by",
    "sys_updated_at",
    "u_symptom",
    "cmdb_ci",
    "knowledge",
    "u_priority_confirmation",
    "notify",
    "problem_id",
    "rfc",
    "vendor",
    "caused_by",
    "closed_code",
    "resolved_by",
]


# ---------------------------------------------------------------------------
# Database connection
# ---------------------------------------------------------------------------

def get_connection() -> psycopg.Connection:
    """
    Create a connection to the configured PostgreSQL database.

    Database credentials are supplied through the DATABASE_URL
    environment variable rather than being stored in source code.
    """

    return psycopg.connect(DATABASE_URL)

# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def validate_input_file() -> None:
    """
    Validate that the cleaned incident dataset exists and has
    the expected analytical schema.
    """

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_FILE}"
        )

    with INPUT_FILE.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        header_line = file.readline().strip()

    actual_columns = header_line.split(",")

    if actual_columns != EXPECTED_COLUMNS:
        raise ValueError(
            "Input CSV schema does not match the expected "
            "incident schema.\n"
            f"Expected columns: {EXPECTED_COLUMNS}\n"
            f"Actual columns:   {actual_columns}"
        )


# ---------------------------------------------------------------------------
# Schema initialization
# ---------------------------------------------------------------------------

def initialize_schema(
    connection: psycopg.Connection,
) -> None:
    """
    Create the incidents table using the project's SQL DDL.
    """

    if not SCHEMA_FILE.exists():
        raise FileNotFoundError(
            f"Schema file not found: {SCHEMA_FILE}"
        )

    schema_sql = SCHEMA_FILE.read_text(
        encoding="utf-8"
    )

    with connection.cursor() as cursor:
        cursor.execute(schema_sql)


# ---------------------------------------------------------------------------
# Bulk loading
# ---------------------------------------------------------------------------
def load_incidents(
    connection: psycopg.Connection,
) -> int:
    """
    Bulk-load incidents_clean.csv into PostgreSQL using COPY.

    Returns
    -------
    int
        Number of rows loaded.
    """

    copy_sql = """
        COPY incidents
        FROM STDIN
        WITH (
            FORMAT CSV,
            HEADER TRUE,
            NULL ''
        )
    """

    with connection.cursor() as cursor:
        cursor.execute(
            "SET datestyle = 'ISO, DMY'"
        )

        with cursor.copy(copy_sql) as copy:
            with INPUT_FILE.open(
                "rb"
            ) as file:
                while chunk := file.read(1024 * 1024):
                    copy.write(chunk)

    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT COUNT(*) FROM incidents"
        )
        row_count = cursor.fetchone()[0]

    return int(row_count)
# ---------------------------------------------------------------------------
# Main loading workflow
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Initialize the analytical database and load incident data.
    """

    print("=" * 70)
    print("IT SERVICE DESK — POSTGRESQL DATA LOADER")
    print("=" * 70)

    print("\n[1/4] Validating input dataset...")
    validate_input_file()
    print(
        f"Input dataset validated: {INPUT_FILE}"
    )

    print("\n[2/4] Connecting to PostgreSQL...")

    with get_connection() as connection:

        print(
            f"Connected to database '{DB_NAME}' "
            f"as '{DB_USER}'."
        )

        print("\n[3/4] Initializing database schema...")
        initialize_schema(connection)

        print("Schema initialized.")

        print("\n[4/4] Loading incident data...")

        row_count = load_incidents(connection)

        expected_row_count = 24_918

        if row_count != expected_row_count:
            raise ValueError(
                "Unexpected incident row count after loading: "
                f"expected {expected_row_count:,}, "
                f"found {row_count:,}"
            )

        print(
            f"Loaded {row_count:,} incident records."
        )

    print("\nDatabase loading completed successfully.")


if __name__ == "__main__":
    main()
