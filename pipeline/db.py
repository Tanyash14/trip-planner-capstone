"""
db.py - duplicated into pipeline/, mcp_server/, and frontend/ (see common/ for the reference copy & duplication note)

NOTE: this file is intentionally duplicated identically into `pipeline/`,
`mcp_server/`, and `frontend/` rather than imported from a shared folder,
because Databricks Apps/Jobs deployed from a Git subdirectory are sandboxed
to that directory and can't import from `../common` (learned the hard way
on the earlier weather-mcp-homework project - see that repo's README). The
canonical reference copy lives in `common/db.py` at the repo root - if you
change the logic here, copy the change into the other two runtime copies too.

Low-level connection + schema layer for the trip planner's Lakebase tables.
Mirrors the `query_log.py` pattern from the weather project: detect the
backend from DATABASE_URL, and provide a small connection context manager
that works identically against SQLite (zero-setup local dev) or
Postgres/Lakebase (production).

Tables (per the capstone spec):
    users, trips, destinations, activities, itinerary_items,
    weather_snapshots, packing_items

Embeddings (for destinations.description and activities.description) are
stored as a JSON-encoded list of floats in a TEXT column rather than a
native pgvector column. This is a deliberate portability choice: it works
identically on SQLite and Postgres/Lakebase without needing the pgvector
extension enabled (which isn't always available/grantable on a managed
Lakebase instance), at the cost of doing similarity search in Python
instead of in the database. See `embeddings.py` for the similarity search
implementation. If you later confirm pgvector is available on your Lakebase
instance, swapping the embedding columns to a native `vector` type and the
Python cosine-similarity loop for an SQL `ORDER BY embedding <=> :query`
is a natural, contained upgrade.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_DEFAULT_SQLITE_PATH = Path(__file__).resolve().parent / "trip_planner.db"
DATABASE_URL = os.environ.get("DATABASE_URL", f"sqlite:///{_DEFAULT_SQLITE_PATH}")
_lock = threading.Lock()


def use_postgres() -> bool:
    return DATABASE_URL.startswith("postgresql")


@contextmanager
def get_connection():
    """Yield (connection, placeholder_style). placeholder_style is '?' for
    SQLite and '%s' for Postgres - use it to build parameterized queries
    that work against either backend."""
    if use_postgres():
        import psycopg2
        import psycopg2.extras

        conn = psycopg2.connect(DATABASE_URL.replace("postgresql+psycopg2://", "postgresql://"))
        try:
            yield conn, "%s"
            conn.commit()
        finally:
            conn.close()
    else:
        sqlite_path = DATABASE_URL.replace("sqlite:///", "")
        conn = sqlite3.connect(sqlite_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn, "?"
            conn.commit()
        finally:
            conn.close()


def _pk(postgres: bool) -> str:
    return "SERIAL PRIMARY KEY" if postgres else "INTEGER PRIMARY KEY AUTOINCREMENT"


def _now(postgres: bool) -> str:
    return "TIMESTAMP DEFAULT CURRENT_TIMESTAMP" if postgres else "TEXT DEFAULT CURRENT_TIMESTAMP"


_SCHEMA_TEMPLATE = """
CREATE TABLE IF NOT EXISTS users (
    id {pk},
    name TEXT NOT NULL,
    email TEXT UNIQUE,
    preferences_text TEXT,           -- free-text interests/notes, e.g. "loves hiking and museums, avoids crowds"
    preferences_embedding TEXT,      -- JSON-encoded float list, see embeddings.py
    created_at {now}
);

CREATE TABLE IF NOT EXISTS trips (
    id {pk},
    user_id INTEGER NOT NULL REFERENCES users(id),
    name TEXT NOT NULL,
    start_date TEXT NOT NULL,        -- ISO date
    end_date TEXT NOT NULL,          -- ISO date
    notes TEXT,
    created_at {now}
);

CREATE TABLE IF NOT EXISTS destinations (
    id {pk},
    trip_id INTEGER NOT NULL REFERENCES trips(id),
    name TEXT NOT NULL,              -- as given by the user, e.g. "Kyoto, Japan"
    resolved_name TEXT,              -- geocoder's canonical name
    latitude REAL,
    longitude REAL,
    timezone TEXT,
    wikipedia_title TEXT,            -- matched Wikipedia article title, if found
    description TEXT,                -- Wikimedia summary text
    description_embedding TEXT,      -- JSON-encoded float list
    created_at {now}
);

CREATE TABLE IF NOT EXISTS activities (
    id {pk},
    destination_id INTEGER NOT NULL REFERENCES destinations(id),
    name TEXT NOT NULL,
    category TEXT,                   -- e.g. "hiking", "museum", "food", "landmark"
    description TEXT,
    description_embedding TEXT,      -- JSON-encoded float list
    is_outdoor INTEGER NOT NULL DEFAULT 0,       -- 0/1 boolean
    requires_good_weather INTEGER NOT NULL DEFAULT 0,  -- 0/1 boolean
    duration_minutes INTEGER,
    source TEXT,                     -- 'wikimedia_attraction' | 'user_added'
    created_at {now}
);

CREATE TABLE IF NOT EXISTS itinerary_items (
    id {pk},
    trip_id INTEGER NOT NULL REFERENCES trips(id),
    destination_id INTEGER NOT NULL REFERENCES destinations(id),
    activity_id INTEGER REFERENCES activities(id),
    scheduled_date TEXT NOT NULL,     -- ISO date
    start_time TEXT,                  -- HH:MM, optional
    end_time TEXT,                    -- HH:MM, optional
    order_index INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'planned',  -- planned | rescheduled | cancelled | completed
    reschedule_reason TEXT,           -- set when the agent moves/cancels this item for weather/AQI reasons
    notes TEXT,
    created_at {now},
    updated_at {now}
);

CREATE TABLE IF NOT EXISTS weather_snapshots (
    id {pk},
    destination_id INTEGER NOT NULL REFERENCES destinations(id),
    date TEXT NOT NULL,               -- ISO date
    high_f REAL,
    low_f REAL,
    precip_chance_pct REAL,
    conditions TEXT,
    aqi REAL,
    pm2_5 REAL,
    pm10 REAL,
    uv_index REAL,
    pollen_index REAL,                -- may be NULL (Open-Meteo pollen coverage is Europe-only)
    fetched_at {now}
);

CREATE TABLE IF NOT EXISTS packing_items (
    id {pk},
    trip_id INTEGER NOT NULL REFERENCES trips(id),
    item_name TEXT NOT NULL,
    category TEXT,                    -- e.g. "clothing", "gear", "documents", "weather"
    quantity INTEGER DEFAULT 1,
    reason TEXT,                      -- why the agent added it, e.g. "rain forecast Aug 12-13"
    packed INTEGER NOT NULL DEFAULT 0,  -- 0/1 boolean
    created_at {now}
);
"""


def init_schema() -> None:
    """Create all tables if they don't already exist. Safe to call on every startup."""
    postgres = use_postgres()
    ddl = _SCHEMA_TEMPLATE.format(pk=_pk(postgres), now=_now(postgres))
    with _lock, get_connection() as (conn, _ph):
        cur = conn.cursor()
        for statement in ddl.split(";"):
            statement = statement.strip()
            if statement:
                cur.execute(statement)


def run_query(sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    """Run a SELECT and return rows as plain dicts."""
    with _lock, get_connection() as (conn, _ph):
        cur = conn.cursor()
        cur.execute(sql, params)
        rows = cur.fetchall()
        columns = [d[0] for d in cur.description] if cur.description else []
        return [dict(zip(columns, row)) for row in rows]


def run_write(sql: str, params: tuple = ()) -> int:
    """Run an INSERT/UPDATE/DELETE. Returns the last inserted row id (INSERTs only,
    best-effort - not all backends return this the same way)."""
    with _lock, get_connection() as (conn, _ph):
        cur = conn.cursor()
        cur.execute(sql, params)
        if use_postgres():
            try:
                return cur.fetchone()[0]  # works if the caller appended "RETURNING id"
            except Exception:
                return cur.rowcount
        return cur.lastrowid
