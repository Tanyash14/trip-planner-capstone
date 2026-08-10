"""
ingest.py

The capstone's "data pipeline" requirement, implemented in plain Python
(no Spark, per the relaxed requirement) so it runs as a Databricks Jobs
Python task (Task type: Python script) against serverless compute, or
locally for testing.

For a given trip, this pipeline:
    1. Geocodes each destination and pulls a Wikipedia summary (Wikimedia API).
    2. Embeds the description and inserts the destination row.
    3. Finds nearby points of interest (Wikimedia geosearch), embeds each,
       and inserts them as `activities` rows - this is the "unstructured
       data processing" requirement: real article text -> embeddings ->
       semantically searchable rows.
    4. Fetches a daily forecast + air quality forecast (Open-Meteo) for the
       destination across the trip's date range and stores it in
       `weather_snapshots`, so the frontend can render a per-day risk
       dashboard without hitting live APIs on every page view.

Usage:
    python ingest.py --trip-id 1 --destinations "Kyoto, Japan" "Osaka, Japan"

    (destinations become new rows on the given trip; run once per trip,
    or again to add more destinations to an existing trip)

As a Databricks Job: set the task type to "Python script", point it at this
file, and pass --trip-id / --destinations as job parameters. No cluster or
Spark runtime needed - a serverless Python environment is enough since this
never touches a Spark DataFrame.
"""

from __future__ import annotations

import argparse
import os
import sys

# NOTE: Lakebase connection details are set as env vars (PGHOST/PGUSER/etc,
# see db.py's _lakebase_connect()) BEFORE importing db/trip_store, because
# this Databricks Jobs workspace's Serverless Python-script task type has no
# UI for setting task-level environment variables (unlike Databricks Apps,
# which get PGHOST etc. auto-injected via "+ Add resource -> Database"). So
# instead we accept them as CLI parameters here and export them into
# os.environ ourselves, right at the top before any Lakebase-touching import
# runs its module-level code.
_conn_parser = argparse.ArgumentParser(add_help=False)
_conn_parser.add_argument("--pghost")
_conn_parser.add_argument("--pgport", default="5432")
_conn_parser.add_argument("--pgdatabase", default="databricks_postgres")
_conn_parser.add_argument("--pgsslmode", default="require")
_conn_parser.add_argument("--pguser")
_conn_parser.add_argument("--endpoint-name")
_conn_args, _ = _conn_parser.parse_known_args()
for _env_name, _val in (
    ("PGHOST", _conn_args.pghost), ("PGPORT", _conn_args.pgport),
    ("PGDATABASE", _conn_args.pgdatabase), ("PGSSLMODE", _conn_args.pgsslmode),
    ("PGUSER", _conn_args.pguser), ("ENDPOINT_NAME", _conn_args.endpoint_name),
):
    if _val:
        os.environ[_env_name] = _val

import db
import open_meteo_broker as weather
import trip_store as store


def ingest_destination(trip_id: int, location_name: str, trip_start: str, trip_end: str) -> None:
    print(f"[ingest] Adding destination '{location_name}' to trip {trip_id}...")
    destination = store.add_destination(trip_id, location_name)
    print(f"[ingest]   -> destination_id={destination['id']} resolved='{destination['resolved_name']}'")

    print(f"[ingest]   Fetching nearby attractions from Wikimedia...")
    activities = store.add_activities_from_attractions(destination["id"])
    print(f"[ingest]   -> added {len(activities)} activities "
          f"({sum(1 for a in activities if a['is_outdoor'])} outdoor, "
          f"{sum(1 for a in activities if not a['is_outdoor'])} indoor)")

    print(f"[ingest]   Fetching weather + air quality forecast ({trip_start} to {trip_end})...")
    try:
        daily = weather.get_daily_forecast(destination["latitude"], destination["longitude"], 16, destination.get("timezone"))
    except weather.WeatherLookupError as exc:
        print(f"[ingest]   ! weather forecast failed: {exc}")
        daily = []

    try:
        air = weather.get_air_quality(destination["latitude"], destination["longitude"], 7, destination.get("timezone"))
    except weather.WeatherLookupError as exc:
        print(f"[ingest]   ! air quality forecast failed: {exc}")
        air = []
    air_by_date = {a["date"]: a for a in air}

    saved = 0
    for day in daily:
        if not (trip_start <= day["date"] <= trip_end):
            continue
        air_day = air_by_date.get(day["date"], {})
        store.save_weather_snapshot(
            destination_id=destination["id"], date=day["date"],
            high_f=day["high_f"], low_f=day["low_f"],
            precip_chance_pct=day["precip_chance_pct"], conditions=day["conditions"],
            aqi=air_day.get("aqi_max"), pm2_5=air_day.get("pm2_5_max"), pm10=air_day.get("pm10_max"),
            uv_index=air_day.get("uv_index_max"), pollen_index=air_day.get("pollen_index_max"),
        )
        saved += 1
    print(f"[ingest]   -> saved {saved} weather_snapshots rows within the trip date range")


def run(trip_id: int, destinations: list[str]) -> None:
    db.init_schema()
    trip = store.get_trip(trip_id)
    print(f"[ingest] Trip {trip_id}: '{trip['name']}' ({trip['start_date']} to {trip['end_date']})")
    for location_name in destinations:
        try:
            ingest_destination(trip_id, location_name, trip["start_date"], trip["end_date"])
        except Exception as exc:  # noqa: BLE001 - a bad destination shouldn't abort the whole job
            print(f"[ingest] ! Failed to ingest '{location_name}': {exc}", file=sys.stderr)
    print("[ingest] Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest destinations, activities, and weather for a trip.",
        parents=[_conn_parser],
    )
    parser.add_argument("--trip-id", type=int, required=True, help="Existing trip id (create it via the frontend first).")
    parser.add_argument("--destinations", nargs="+", required=True, help="One or more destination names, e.g. \"Kyoto, Japan\" \"Osaka, Japan\".")
    args = parser.parse_args()
    run(args.trip_id, args.destinations)


if __name__ == "__main__":
    main()
