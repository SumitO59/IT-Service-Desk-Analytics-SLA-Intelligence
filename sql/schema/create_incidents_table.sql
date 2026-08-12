DROP TABLE IF EXISTS incidents;

CREATE TABLE incidents (
    number TEXT PRIMARY KEY,

    opened_at TIMESTAMP,
    resolved_at TIMESTAMP,
    closed_at TIMESTAMP,

    resolution_time_hours DOUBLE PRECISION,
    closure_time_hours DOUBLE PRECISION,

    made_sla BOOLEAN,
    sla_breached BOOLEAN,

    reassignment_count INTEGER,
    reassignment_bucket TEXT,

    reopen_count INTEGER,
    reopened BOOLEAN,

    sys_mod_count INTEGER,

    incident_state TEXT,
    active BOOLEAN,

    category TEXT,
    subcategory TEXT,
    priority TEXT,
    impact TEXT,
    urgency TEXT,

    assignment_group TEXT,
    assigned_to TEXT,
    caller_id TEXT,

    contact_type TEXT,
    location TEXT,

    opened_date DATE,
    opened_month TEXT,
    opened_year INTEGER,

    opened_by TEXT,
    sys_created_by TEXT,
    sys_created_at TIMESTAMP,
    sys_updated_by TEXT,
    sys_updated_at TIMESTAMP,

    u_symptom TEXT,
    cmdb_ci TEXT,

    knowledge BOOLEAN,
    u_priority_confirmation BOOLEAN,

    notify TEXT,
    problem_id TEXT,
    rfc TEXT,
    vendor TEXT,
    caused_by TEXT,
    closed_code TEXT,
    resolved_by TEXT
);
