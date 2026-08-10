"""
trip_store.py (reference copy - see note in db.py about duplication)

The data-access layer for the trip planner's Lakebase tables - plays the
same role for the database that `open_meteo_broker.py`/`wikimedia_broker.py`
play for HTTP APIs: all SQL lives here, callers get clean dicts, and MCP
tool functions stay thin wrappers around these functions.

Organized in four sections:
    1. Users & trips
    2. Destinations & activities (enrichment + semantic search)
    3. Itinerary items (CRUD + the deterministic reschedule-check logic)
    4. Packing items

The reschedule-check logic (`check_reschedule_needed`) is the "show your
reasoning, don't just pass through raw data" piece, in the same spirit as
`predict_umbrella_needed` from the earlier weather project: explicit
thresholds, applied to real fetched data, returning a reason string the
agent can quote directly instead of inventing one.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import db
import embeddings
import open_meteo_broker as weather
import wikimedia_broker as wiki

# --- Thresholds for the deterministic reschedule/health-flag logic --------
RAIN_RESCHEDULE_THRESHOLD_PCT = 40      # matches the weather project's umbrella threshold
AQI_UNHEALTHY_THRESHOLD = 150           # US AQI "Unhealthy for Sensitive Groups" starts at 101; 150 = "Unhealthy"
AQI_SENSITIVE_THRESHOLD = 101           # flagged for users with a declared health sensitivity
UV_HIGH_THRESHOLD = 8                   # US EPA "Very High" UV index starts at 8
POLLEN_HIGH_THRESHOLD = 4               # Open-Meteo pollen index is roughly 0-6+ per species; 4+ is notably high


class TripStoreError(Exception):
    """Raised for not-found / invalid-input cases. Callers (MCP tools) catch
    this and return a clean {"error": ...} dict."""


# ---------------------------------------------------------------------------
# 1. Users & trips
# ---------------------------------------------------------------------------

def get_or_create_user(name: str, email: str | None = None) -> dict[str, Any]:
    """Look up a user by email (if given) or name, creating one if none exists."""
    if email:
        rows = db.run_query("SELECT * FROM users WHERE email = ?" if not db.use_postgres() else "SELECT * FROM users WHERE email = %s", (email,))
        if rows:
            return rows[0]
    placeholder = "%s" if db.use_postgres() else "?"
    returning = " RETURNING id" if db.use_postgres() else ""
    new_id = db.run_write(
        f"INSERT INTO users (name, email) VALUES ({placeholder}, {placeholder}){returning}",
        (name, email),
    )
    return get_user(new_id)


def get_user(user_id: int) -> dict[str, Any]:
    ph = "%s" if db.use_postgres() else "?"
    rows = db.run_query(f"SELECT * FROM users WHERE id = {ph}", (user_id,))
    if not rows:
        raise TripStoreError(f"No user found with id {user_id}.")
    return rows[0]


def update_user_preferences(user_id: int, preferences_text: str) -> dict[str, Any]:
    """Save a user's free-text interests/health-notes and embed them for
    semantic activity matching (the 'vibe matching' feature)."""
    vector = embeddings.embed_text(preferences_text)
    ph = "%s" if db.use_postgres() else "?"
    db.run_write(
        f"UPDATE users SET preferences_text = {ph}, preferences_embedding = {ph} WHERE id = {ph}",
        (preferences_text, embeddings.to_json(vector), user_id),
    )
    return get_user(user_id)


def create_trip(user_id: int, name: str, start_date: str, end_date: str, notes: str | None = None) -> dict[str, Any]:
    ph = "%s" if db.use_postgres() else "?"
    returning = " RETURNING id" if db.use_postgres() else ""
    new_id = db.run_write(
        f"INSERT INTO trips (user_id, name, start_date, end_date, notes) VALUES ({ph}, {ph}, {ph}, {ph}, {ph}){returning}",
        (user_id, name, start_date, end_date, notes),
    )
    return get_trip(new_id)


def get_trip(trip_id: int) -> dict[str, Any]:
    ph = "%s" if db.use_postgres() else "?"
    rows = db.run_query(f"SELECT * FROM trips WHERE id = {ph}", (trip_id,))
    if not rows:
        raise TripStoreError(f"No trip found with id {trip_id}.")
    return rows[0]


def list_trips(user_id: int) -> list[dict[str, Any]]:
    ph = "%s" if db.use_postgres() else "?"
    return db.run_query(f"SELECT * FROM trips WHERE user_id = {ph} ORDER BY start_date", (user_id,))


# ---------------------------------------------------------------------------
# 2. Destinations & activities
# ---------------------------------------------------------------------------

def add_destination(trip_id: int, location_name: str) -> dict[str, Any]:
    """Add a destination to a trip: geocodes it, pulls a Wikipedia summary,
    embeds the description, and inserts the row. This is the same
    enrichment logic the ingestion pipeline uses, exposed here so the agent
    can also add destinations conversationally ("add Kyoto to my trip")."""
    place = weather.geocode_location(location_name)
    summary = wiki.get_destination_summary(location_name)
    description = summary.get("extract") or ""
    vector = embeddings.embed_text(description) if description else []

    ph = "%s" if db.use_postgres() else "?"
    returning = " RETURNING id" if db.use_postgres() else ""
    new_id = db.run_write(
        f"""INSERT INTO destinations
            (trip_id, name, resolved_name, latitude, longitude, timezone, wikipedia_title, description, description_embedding)
            VALUES ({ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph}){returning}""",
        (trip_id, location_name, _format_place(place), place["latitude"], place["longitude"],
         place.get("timezone"), summary.get("title"), description,
         embeddings.to_json(vector) if vector else None),
    )
    return get_destination(new_id)


def get_destination(destination_id: int) -> dict[str, Any]:
    ph = "%s" if db.use_postgres() else "?"
    rows = db.run_query(f"SELECT * FROM destinations WHERE id = {ph}", (destination_id,))
    if not rows:
        raise TripStoreError(f"No destination found with id {destination_id}.")
    return rows[0]


def list_destinations(trip_id: int) -> list[dict[str, Any]]:
    ph = "%s" if db.use_postgres() else "?"
    return db.run_query(f"SELECT * FROM destinations WHERE trip_id = {ph} ORDER BY id", (trip_id,))


def add_activities_from_attractions(destination_id: int, radius_m: int = 10000, limit: int = 15) -> list[dict[str, Any]]:
    """Populate `activities` for a destination from nearby Wikipedia points
    of interest (the ingestion pipeline's main enrichment step, also
    callable directly). Each attraction is embedded for semantic search and
    tagged `source='wikimedia_attraction'`. Outdoor/weather-sensitivity is
    guessed from simple keyword heuristics in the title/extract - imperfect,
    but a reasonable default that users/the agent can correct via
    `add_activity` or a future edit tool."""
    destination = get_destination(destination_id)
    attractions = wiki.get_nearby_attractions(destination["latitude"], destination["longitude"], radius_m, limit)

    inserted = []
    for attraction in attractions:
        name = attraction["title"]
        description = attraction["extract"] or ""
        is_outdoor, category = _guess_category(name, description)
        vector = embeddings.embed_text(description) if description else []
        row = _insert_activity(
            destination_id=destination_id, name=name, category=category, description=description,
            embedding_vector=vector, is_outdoor=is_outdoor, requires_good_weather=is_outdoor,
            duration_minutes=90, source="wikimedia_attraction",
        )
        inserted.append(row)
    return inserted


def add_activity(destination_id: int, name: str, category: str | None = None, description: str | None = None,
                  is_outdoor: bool = False, requires_good_weather: bool = False,
                  duration_minutes: int | None = None) -> dict[str, Any]:
    """Add a user-defined activity (not sourced from Wikipedia)."""
    vector = embeddings.embed_text(description) if description else []
    return _insert_activity(
        destination_id=destination_id, name=name, category=category, description=description or "",
        embedding_vector=vector, is_outdoor=is_outdoor, requires_good_weather=requires_good_weather,
        duration_minutes=duration_minutes, source="user_added",
    )


def _insert_activity(destination_id, name, category, description, embedding_vector, is_outdoor,
                      requires_good_weather, duration_minutes, source) -> dict[str, Any]:
    ph = "%s" if db.use_postgres() else "?"
    returning = " RETURNING id" if db.use_postgres() else ""
    new_id = db.run_write(
        f"""INSERT INTO activities
            (destination_id, name, category, description, description_embedding, is_outdoor, requires_good_weather, duration_minutes, source)
            VALUES ({ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph}){returning}""",
        (destination_id, name, category, description,
         embeddings.to_json(embedding_vector) if embedding_vector else None,
         1 if is_outdoor else 0, 1 if requires_good_weather else 0, duration_minutes, source),
    )
    rows = db.run_query(f"SELECT * FROM activities WHERE id = {ph}", (new_id,))
    return rows[0]


def search_activities(destination_id: int, query_text: str, top_k: int = 5, outdoor_only: bool | None = None) -> list[dict[str, Any]]:
    """Semantic search over a destination's activities using embedding
    cosine similarity ('vibe matching') rather than keyword filtering.

    Args:
        destination_id: destination to search within.
        query_text: free-text interest/vibe description, e.g. "quiet nature spots, not touristy".
        top_k: max results to return.
        outdoor_only: if set, filter to only outdoor (True) or only indoor (False) activities first.

    Returns:
        list of activity dicts, each with a `similarity` score (0-1, higher = better match).
    """
    ph = "%s" if db.use_postgres() else "?"
    sql = f"SELECT * FROM activities WHERE destination_id = {ph}"
    params: list[Any] = [destination_id]
    if outdoor_only is not None:
        sql += f" AND is_outdoor = {ph}"
        params.append(1 if outdoor_only else 0)
    candidates = db.run_query(sql, tuple(params))
    if not candidates:
        return []

    query_vector = embeddings.embed_text(query_text)
    return embeddings.rank_by_similarity(query_vector, candidates, embedding_key="description_embedding", top_k=top_k)


def search_destinations(trip_id: int, query_text: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Semantic search over a trip's destinations (useful once a trip has
    several legs and the user asks something like 'which stop had the old
    temples?')."""
    ph = "%s" if db.use_postgres() else "?"
    candidates = db.run_query(f"SELECT * FROM destinations WHERE trip_id = {ph}", (trip_id,))
    if not candidates:
        return []
    query_vector = embeddings.embed_text(query_text)
    return embeddings.rank_by_similarity(query_vector, candidates, embedding_key="description_embedding", top_k=top_k)


def _guess_category(name: str, description: str) -> tuple[bool, str]:
    """Cheap keyword heuristic for whether a Wikipedia POI is outdoor and
    what category it roughly falls into. Good-enough default for
    auto-ingested attractions; not a substitute for real classification."""
    text = f"{name} {description}".lower()
    outdoor_keywords = ["park", "garden", "trail", "mountain", "beach", "hike", "hiking", "lake",
                         "river", "forest", "trek", "waterfall", "peak", "bridge", "square", "plaza"]
    museum_keywords = ["museum", "gallery", "exhibit"]
    food_keywords = ["market", "food", "restaurant", "cuisine"]
    religious_keywords = ["temple", "shrine", "church", "cathedral", "mosque"]

    is_outdoor = any(k in text for k in outdoor_keywords)
    if any(k in text for k in museum_keywords):
        category = "museum"
    elif any(k in text for k in food_keywords):
        category = "food"
    elif any(k in text for k in religious_keywords):
        category = "landmark"
    elif is_outdoor:
        category = "outdoor"
    else:
        category = "sightseeing"
    return is_outdoor, category


def _format_place(place: dict[str, Any]) -> str:
    parts = [p for p in (place.get("name"), place.get("admin1"), place.get("country")) if p]
    return ", ".join(parts) if parts else f"{place['latitude']},{place['longitude']}"


def save_weather_snapshot(destination_id: int, date: str, high_f: float | None, low_f: float | None,
                           precip_chance_pct: float | None, conditions: str | None,
                           aqi: float | None = None, pm2_5: float | None = None, pm10: float | None = None,
                           uv_index: float | None = None, pollen_index: float | None = None) -> dict[str, Any]:
    """Store a day's weather/AQI numbers for a destination - used by the
    ingestion pipeline to pre-populate `weather_snapshots` so the frontend
    can show a per-day risk dashboard without re-calling the live APIs on
    every page load. `check_reschedule_needed` still calls the live APIs
    directly for up-to-date numbers when the agent is actually making a
    reschedule decision."""
    ph = "%s" if db.use_postgres() else "?"
    returning = " RETURNING id" if db.use_postgres() else ""
    new_id = db.run_write(
        f"""INSERT INTO weather_snapshots
            (destination_id, date, high_f, low_f, precip_chance_pct, conditions, aqi, pm2_5, pm10, uv_index, pollen_index)
            VALUES ({ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph}){returning}""",
        (destination_id, date, high_f, low_f, precip_chance_pct, conditions, aqi, pm2_5, pm10, uv_index, pollen_index),
    )
    rows = db.run_query(f"SELECT * FROM weather_snapshots WHERE id = {ph}", (new_id,))
    return rows[0]


def get_weather_snapshots(destination_id: int) -> list[dict[str, Any]]:
    ph = "%s" if db.use_postgres() else "?"
    return db.run_query(f"SELECT * FROM weather_snapshots WHERE destination_id = {ph} ORDER BY date", (destination_id,))


# ---------------------------------------------------------------------------
# 3. Itinerary items
# ---------------------------------------------------------------------------

def get_itinerary(trip_id: int) -> list[dict[str, Any]]:
    """Full itinerary for a trip, joined with activity/destination names,
    ordered by date then order_index. Raises TripStoreError for a
    nonexistent trip_id rather than silently returning an empty list, so
    the agent can distinguish "no items yet" from "that trip doesn't exist"."""
    get_trip(trip_id)  # raises TripStoreError if the trip doesn't exist
    ph = "%s" if db.use_postgres() else "?"
    return db.run_query(f"""
        SELECT ii.*, a.name AS activity_name, a.category AS activity_category,
               a.is_outdoor, a.requires_good_weather, d.name AS destination_name
        FROM itinerary_items ii
        LEFT JOIN activities a ON ii.activity_id = a.id
        JOIN destinations d ON ii.destination_id = d.id
        WHERE ii.trip_id = {ph}
        ORDER BY ii.scheduled_date, ii.order_index
    """, (trip_id,))


def add_itinerary_item(trip_id: int, destination_id: int, scheduled_date: str, activity_id: int | None = None,
                        start_time: str | None = None, end_time: str | None = None, notes: str | None = None) -> dict[str, Any]:
    ph = "%s" if db.use_postgres() else "?"
    existing = db.run_query(f"SELECT COALESCE(MAX(order_index), -1) AS max_idx FROM itinerary_items WHERE trip_id = {ph} AND scheduled_date = {ph}", (trip_id, scheduled_date))
    next_idx = (existing[0]["max_idx"] if existing else -1) + 1
    returning = " RETURNING id" if db.use_postgres() else ""
    new_id = db.run_write(
        f"""INSERT INTO itinerary_items
            (trip_id, destination_id, activity_id, scheduled_date, start_time, end_time, order_index, notes)
            VALUES ({ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph}){returning}""",
        (trip_id, destination_id, activity_id, scheduled_date, start_time, end_time, next_idx, notes),
    )
    return get_itinerary_item(new_id)


def get_itinerary_item(item_id: int) -> dict[str, Any]:
    ph = "%s" if db.use_postgres() else "?"
    rows = db.run_query(f"SELECT * FROM itinerary_items WHERE id = {ph}", (item_id,))
    if not rows:
        raise TripStoreError(f"No itinerary item found with id {item_id}.")
    return rows[0]


def move_itinerary_item(item_id: int, new_date: str | None = None, new_start_time: str | None = None, new_end_time: str | None = None) -> dict[str, Any]:
    """Move an item to a new date/time without implying a weather-driven
    reschedule (use `reschedule_itinerary_item` for that, which also records
    a reason)."""
    item = get_itinerary_item(item_id)
    ph = "%s" if db.use_postgres() else "?"
    db.run_write(
        f"UPDATE itinerary_items SET scheduled_date = {ph}, start_time = {ph}, end_time = {ph} WHERE id = {ph}",
        (new_date or item["scheduled_date"], new_start_time, new_end_time, item_id),
    )
    return get_itinerary_item(item_id)


def reschedule_itinerary_item(item_id: int, new_date: str, reason: str) -> dict[str, Any]:
    """Move an item to a new date because of a weather/AQI-driven decision,
    recording the reason so the frontend's change-log can show it and the
    agent can quote it verbatim rather than re-explaining from memory."""
    ph = "%s" if db.use_postgres() else "?"
    db.run_write(
        f"UPDATE itinerary_items SET scheduled_date = {ph}, status = 'rescheduled', reschedule_reason = {ph} WHERE id = {ph}",
        (new_date, reason, item_id),
    )
    return get_itinerary_item(item_id)


def remove_itinerary_item(item_id: int) -> dict[str, Any]:
    item = get_itinerary_item(item_id)
    ph = "%s" if db.use_postgres() else "?"
    db.run_write(f"DELETE FROM itinerary_items WHERE id = {ph}", (item_id,))
    return item


def check_reschedule_needed(item_id: int) -> dict[str, Any]:
    """Deterministic go/no-go check for a single itinerary item, combining
    live forecast + air quality data with the item's outdoor/weather-
    sensitivity flags and the trip's declared health sensitivities. This is
    the grounding logic behind the agent's "explain why" requirement -
    thresholds are explicit and applied to real fetched numbers, not left
    to the LLM to eyeball.

    Thresholds applied:
        - Rain: reschedule if precip_chance_pct > 40 AND the activity is
          outdoor/weather-sensitive.
        - Air quality: flag if AQI > 150 (unhealthy for everyone), or > 101
          when the trip's user has a declared health sensitivity in their
          preferences_text (asthma, allergies, respiratory, etc.).
        - UV: flag (not reschedule) if UV index > 8 for outdoor activities -
          surfaced as a packing/precaution note, not a reschedule reason.
        - Pollen: flag if pollen index > 4 and the user has an allergy note.

    Returns:
        dict with item_id, needs_reschedule (bool), reasons (list of str),
        precautions (list of str - non-reschedule warnings like UV/pollen),
        and the raw weather/aqi numbers used, so the agent can cite them.
    """
    item = get_itinerary_item(item_id)
    destination = get_destination(item["destination_id"])
    ph = "%s" if db.use_postgres() else "?"
    activity_rows = db.run_query(f"SELECT * FROM activities WHERE id = {ph}", (item["activity_id"],)) if item["activity_id"] else []
    activity = activity_rows[0] if activity_rows else None
    is_outdoor = bool(activity["is_outdoor"]) if activity else False
    weather_sensitive = bool(activity["requires_good_weather"]) if activity else False

    trip = get_trip(item["trip_id"])
    user = get_user(trip["user_id"])
    health_sensitive = _has_health_sensitivity(user.get("preferences_text") or "")

    daily = weather.get_daily_forecast(destination["latitude"], destination["longitude"], 16, destination.get("timezone"))
    day_weather = next((d for d in daily if d["date"] == item["scheduled_date"]), None)

    try:
        air = weather.get_air_quality(destination["latitude"], destination["longitude"], 7, destination.get("timezone"))
        day_air = next((d for d in air if d["date"] == item["scheduled_date"]), None)
    except weather.WeatherLookupError:
        day_air = None  # air quality forecast window is shorter than weather; not fatal

    reasons, precautions = [], []
    needs_reschedule = False

    if day_weather and (is_outdoor or weather_sensitive):
        pct = day_weather["precip_chance_pct"] or 0
        if pct > RAIN_RESCHEDULE_THRESHOLD_PCT:
            needs_reschedule = True
            reasons.append(f"{pct}% chance of rain ({day_weather['conditions']}) on {item['scheduled_date']}, above the {RAIN_RESCHEDULE_THRESHOLD_PCT}% threshold for an outdoor activity.")

    if day_air and (is_outdoor or weather_sensitive):
        aqi = day_air.get("aqi_max")
        if aqi is not None:
            if aqi > AQI_UNHEALTHY_THRESHOLD:
                needs_reschedule = True
                reasons.append(f"Air quality forecast is {aqi:.0f} AQI (Unhealthy) on {item['scheduled_date']}, above the {AQI_UNHEALTHY_THRESHOLD} threshold.")
            elif aqi > AQI_SENSITIVE_THRESHOLD and health_sensitive:
                needs_reschedule = True
                reasons.append(f"Air quality forecast is {aqi:.0f} AQI (Unhealthy for Sensitive Groups) on {item['scheduled_date']} - flagged because your profile notes a health sensitivity.")
        uv = day_air.get("uv_index_max")
        if uv is not None and uv > UV_HIGH_THRESHOLD and is_outdoor:
            precautions.append(f"UV index forecast is {uv:.1f} (Very High) on {item['scheduled_date']} - sunscreen/hat/shade breaks recommended.")
        pollen = day_air.get("pollen_index_max")
        if pollen is not None and pollen > POLLEN_HIGH_THRESHOLD and health_sensitive:
            precautions.append(f"Pollen index forecast is {pollen:.1f} on {item['scheduled_date']} - consider allergy medication if sensitive.")

    return {
        "item_id": item_id, "scheduled_date": item["scheduled_date"],
        "needs_reschedule": needs_reschedule, "reasons": reasons, "precautions": precautions,
        "weather": day_weather, "air_quality": day_air,
        "is_outdoor": is_outdoor, "requires_good_weather": weather_sensitive,
    }


def _has_health_sensitivity(preferences_text: str) -> bool:
    keywords = ["asthma", "allerg", "respiratory", "copd", "lung", "sensitive to pollen", "sensitive to air"]
    text = preferences_text.lower()
    return any(k in text for k in keywords)


# ---------------------------------------------------------------------------
# 4. Packing items
# ---------------------------------------------------------------------------

def get_packing_list(trip_id: int) -> list[dict[str, Any]]:
    get_trip(trip_id)  # raises TripStoreError if the trip doesn't exist
    ph = "%s" if db.use_postgres() else "?"
    return db.run_query(f"SELECT * FROM packing_items WHERE trip_id = {ph} ORDER BY category, item_name", (trip_id,))


def add_packing_item(trip_id: int, item_name: str, category: str | None = None, quantity: int = 1, reason: str | None = None) -> dict[str, Any]:
    ph = "%s" if db.use_postgres() else "?"
    returning = " RETURNING id" if db.use_postgres() else ""
    new_id = db.run_write(
        f"INSERT INTO packing_items (trip_id, item_name, category, quantity, reason) VALUES ({ph},{ph},{ph},{ph},{ph}){returning}",
        (trip_id, item_name, category, quantity, reason),
    )
    rows = db.run_query(f"SELECT * FROM packing_items WHERE id = {ph}", (new_id,))
    return rows[0]


def remove_packing_item(item_id: int) -> None:
    ph = "%s" if db.use_postgres() else "?"
    db.run_write(f"DELETE FROM packing_items WHERE id = {ph}", (item_id,))


def set_packed(item_id: int, packed: bool) -> dict[str, Any]:
    ph = "%s" if db.use_postgres() else "?"
    db.run_write(f"UPDATE packing_items SET packed = {ph} WHERE id = {ph}", (1 if packed else 0, item_id))
    rows = db.run_query(f"SELECT * FROM packing_items WHERE id = {ph}", (item_id,))
    return rows[0]
