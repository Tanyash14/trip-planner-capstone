"""
trip_mcp_server.py

FastMCP server exposing the trip planner's agent tools, following the same
streamable-HTTP pattern as the weather-mcp-homework project. Tool functions
are thin: they validate/shape input, delegate all logic to `trip_store.py`
(database) and `open_meteo_broker.py` (live weather/AQI), and translate
failures into clean {"error": ...} dicts instead of raising.

Tools are grouped into:
    - Trip/user setup (write)
    - Destinations & activities - retrieval via semantic search, and
      write tools to add destinations/activities (unstructured-data +
      embeddings requirement)
    - Weather & air quality (read) - live, not cached, so reschedule
      decisions use current data
    - Itinerary items (read + write) - the actual CRUD + reschedule logic
    - Packing list (read + write)

Run locally:
    python trip_mcp_server.py
    # -> streamable-HTTP server on 0.0.0.0:8000/mcp
"""

from __future__ import annotations

import os
from typing import Any

from fastmcp import FastMCP

import db
import open_meteo_broker as weather
import trip_store as store

db.init_schema()  # safe to call on every startup - CREATE TABLE IF NOT EXISTS

mcp = FastMCP("trip-planner-server")


def _err(exc: Exception) -> dict[str, Any]:
    return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Trip / user setup
# ---------------------------------------------------------------------------
@mcp.tool
def create_trip(user_name: str, trip_name: str, start_date: str, end_date: str,
                 user_email: str | None = None, notes: str | None = None) -> dict[str, Any]:
    """Create a new trip for a user (creating the user if they don't already exist).

    Args:
        user_name: the traveler's name.
        trip_name: a short name for the trip, e.g. "Japan Trip 2026".
        start_date: trip start date, YYYY-MM-DD.
        end_date: trip end date, YYYY-MM-DD.
        user_email: optional - used to look up/deduplicate an existing user.
        notes: optional free-text trip notes.

    Returns:
        On success: dict with user (id, name) and trip (id, name, start_date, end_date).
        On failure: dict with a single "error" key.
    """
    try:
        user = store.get_or_create_user(user_name, user_email)
        trip = store.create_trip(user["id"], trip_name, start_date, end_date, notes)
        return {"user": user, "trip": trip}
    except store.TripStoreError as exc:
        return _err(exc)


@mcp.tool
def update_traveler_preferences(user_id: int, preferences_text: str) -> dict[str, Any]:
    """Save a traveler's free-text interests and/or health notes (e.g. allergies,
    asthma), and embed them for semantic activity matching ("vibe matching").
    Health-sensitivity keywords in this text (asthma, allergy, respiratory, etc.)
    also lower the air-quality reschedule threshold for this user's trips.

    Args:
        user_id: the traveler's user id (from create_trip's response).
        preferences_text: free-text description, e.g. "loves hiking and quiet
            nature spots, not into big crowds, has asthma so sensitive to air quality".

    Returns:
        On success: the updated user dict. On failure: dict with an "error" key.
    """
    try:
        return store.update_user_preferences(user_id, preferences_text)
    except store.TripStoreError as exc:
        return _err(exc)


# ---------------------------------------------------------------------------
# Destinations & activities
# ---------------------------------------------------------------------------
@mcp.tool
def add_destination(trip_id: int, location_name: str, populate_activities: bool = True) -> dict[str, Any]:
    """Add a destination to a trip. Geocodes the location, pulls a Wikipedia
    summary, embeds the description for semantic search, and (by default)
    also populates nearby points of interest as candidate activities.

    Args:
        trip_id: the trip to add this destination to.
        location_name: free-text destination name, e.g. "Kyoto, Japan".
        populate_activities: if True (default), also fetch and add nearby
            Wikipedia points of interest as activities in the same call.

    Returns:
        On success: dict with the destination row and, if populate_activities,
        an "activities_added" count. On failure: dict with an "error" key.
    """
    try:
        destination = store.add_destination(trip_id, location_name)
        result: dict[str, Any] = {"destination": destination}
        if populate_activities:
            activities = store.add_activities_from_attractions(destination["id"])
            result["activities_added"] = len(activities)
        return result
    except (store.TripStoreError, weather.WeatherLookupError) as exc:
        return _err(exc)


@mcp.tool
def add_activity(destination_id: int, name: str, description: str = "", category: str | None = None,
                  is_outdoor: bool = False, requires_good_weather: bool = False,
                  duration_minutes: int | None = None) -> dict[str, Any]:
    """Add a custom activity to a destination (not sourced from Wikipedia) -
    e.g. a specific restaurant reservation or tour the traveler requested.

    Args:
        destination_id: the destination this activity belongs to.
        name: activity name.
        description: free-text description (embedded for semantic search).
        category: e.g. "hiking", "museum", "food", "landmark".
        is_outdoor: whether this activity happens outdoors.
        requires_good_weather: whether rain/poor air quality should trigger a reschedule check.
        duration_minutes: expected duration, if known.

    Returns:
        On success: the new activity dict. On failure: dict with an "error" key.
    """
    try:
        return store.add_activity(destination_id, name, category, description, is_outdoor, requires_good_weather, duration_minutes)
    except store.TripStoreError as exc:
        return _err(exc)


@mcp.tool
def search_activities(destination_id: int, interests: str, top_k: int = 5, outdoor_only: bool | None = None) -> dict[str, Any]:
    """Semantically search a destination's activities by free-text interests
    or "vibe" (not a keyword filter - uses embedding similarity), e.g.
    "quiet nature spots, not touristy" or "family-friendly indoor activities".

    Args:
        destination_id: destination to search within.
        interests: free-text description of what the traveler wants.
        top_k: max results to return (default 5).
        outdoor_only: optionally restrict to outdoor (True) or indoor (False) activities first.

    Returns:
        On success: dict with a "results" list, each activity plus a
        "similarity" score (0-1, higher = better match). On failure: dict with an "error" key.
    """
    try:
        results = store.search_activities(destination_id, interests, top_k, outdoor_only)
        return {"results": results}
    except store.TripStoreError as exc:
        return _err(exc)


# ---------------------------------------------------------------------------
# Weather & air quality (live)
# ---------------------------------------------------------------------------
@mcp.tool
def get_forecast(destination_id: int, days: int = 7) -> dict[str, Any]:
    """Get the live daily weather forecast for a destination already on a trip.

    Args:
        destination_id: destination id (must already exist - use add_destination first).
        days: number of days to forecast, 1-16 (default 7).

    Returns:
        On success: dict with resolved_location and a "days" list (date,
        high_f, low_f, precip_chance_pct, conditions). On failure: dict with an "error" key.
    """
    try:
        destination = store.get_destination(destination_id)
        days_data = weather.get_daily_forecast(destination["latitude"], destination["longitude"], days, destination.get("timezone"))
        return {"resolved_location": destination["resolved_name"], "days": days_data}
    except (store.TripStoreError, weather.WeatherLookupError) as exc:
        return _err(exc)


@mcp.tool
def get_air_quality(destination_id: int, days: int = 7) -> dict[str, Any]:
    """Get the live daily air-quality forecast (AQI, PM2.5, PM10, UV, pollen)
    for a destination already on a trip.

    Args:
        destination_id: destination id.
        days: number of days to forecast, 1-7 (default 7).

    Returns:
        On success: dict with resolved_location and a "days" list (date,
        aqi_max, pm2_5_max, pm10_max, uv_index_max, pollen_index_max - pollen
        may be null outside Europe). On failure: dict with an "error" key.
    """
    try:
        destination = store.get_destination(destination_id)
        days_data = weather.get_air_quality(destination["latitude"], destination["longitude"], days, destination.get("timezone"))
        return {"resolved_location": destination["resolved_name"], "days": days_data}
    except (store.TripStoreError, weather.WeatherLookupError) as exc:
        return _err(exc)


# ---------------------------------------------------------------------------
# Itinerary items
# ---------------------------------------------------------------------------
@mcp.tool
def get_itinerary(trip_id: int) -> dict[str, Any]:
    """Get the full day-by-day itinerary for a trip, ordered by date.

    Args:
        trip_id: the trip id.

    Returns:
        On success: dict with an "items" list (scheduled_date, start_time,
        end_time, activity_name, destination_name, status, reschedule_reason).
        On failure: dict with an "error" key.
    """
    try:
        return {"items": store.get_itinerary(trip_id)}
    except store.TripStoreError as exc:
        return _err(exc)


@mcp.tool
def add_itinerary_item(trip_id: int, destination_id: int, scheduled_date: str, activity_id: int | None = None,
                        start_time: str | None = None, end_time: str | None = None, notes: str | None = None) -> dict[str, Any]:
    """Add an item to the itinerary (schedule an activity on a specific date).

    Args:
        trip_id: the trip id.
        destination_id: which destination this item is at.
        scheduled_date: YYYY-MM-DD.
        activity_id: optional - link to a specific activity (from search_activities or add_activity).
        start_time: optional, HH:MM.
        end_time: optional, HH:MM.
        notes: optional free-text notes.

    Returns:
        On success: the new itinerary item dict. On failure: dict with an "error" key.
    """
    try:
        return store.add_itinerary_item(trip_id, destination_id, scheduled_date, activity_id, start_time, end_time, notes)
    except store.TripStoreError as exc:
        return _err(exc)


@mcp.tool
def move_itinerary_item(item_id: int, new_date: str | None = None, new_start_time: str | None = None, new_end_time: str | None = None) -> dict[str, Any]:
    """Move an itinerary item to a new date/time at the traveler's request
    (not a weather-driven change - use reschedule_itinerary_item for that,
    which records a reason).

    Args:
        item_id: the itinerary item id.
        new_date: optional new date, YYYY-MM-DD (unchanged if omitted).
        new_start_time: optional new start time, HH:MM.
        new_end_time: optional new end time, HH:MM.

    Returns:
        On success: the updated itinerary item dict. On failure: dict with an "error" key.
    """
    try:
        return store.move_itinerary_item(item_id, new_date, new_start_time, new_end_time)
    except store.TripStoreError as exc:
        return _err(exc)


@mcp.tool
def remove_itinerary_item(item_id: int) -> dict[str, Any]:
    """Remove an item from the itinerary entirely.

    Args:
        item_id: the itinerary item id.

    Returns:
        On success: the removed item's data (for confirmation). On failure: dict with an "error" key.
    """
    try:
        return store.remove_itinerary_item(item_id)
    except store.TripStoreError as exc:
        return _err(exc)


@mcp.tool
def check_reschedule_needed(item_id: int) -> dict[str, Any]:
    """Check whether an itinerary item should be rescheduled, using live
    weather/air-quality data and explicit thresholds - this is the grounded
    reasoning step, not a guess. Always call this BEFORE calling
    reschedule_itinerary_item, and quote its "reasons" verbatim when
    explaining the change to the traveler.

    Thresholds applied: rain reschedule if precipitation chance > 40% for an
    outdoor/weather-sensitive activity; air-quality reschedule if AQI > 150
    for everyone, or AQI > 101 if the traveler has a declared health
    sensitivity; UV > 8 and pollen > 4 (for sensitive travelers) are
    surfaced as non-reschedule "precautions" instead (e.g. sunscreen, allergy meds).

    Args:
        item_id: the itinerary item id to check.

    Returns:
        On success: dict with needs_reschedule (bool), reasons (list of str,
        empty if no reschedule needed), precautions (list of str), and the
        raw weather/air_quality numbers used. On failure: dict with an "error" key.
    """
    try:
        return store.check_reschedule_needed(item_id)
    except (store.TripStoreError, weather.WeatherLookupError) as exc:
        return _err(exc)


@mcp.tool
def reschedule_itinerary_item(item_id: int, new_date: str, reason: str) -> dict[str, Any]:
    """Move an itinerary item to a new date because of a weather/air-quality
    decision, recording the reason. Call check_reschedule_needed first and
    pass its "reasons" (joined into one string) as the reason argument here -
    never invent a reason that wasn't returned by check_reschedule_needed.

    Args:
        item_id: the itinerary item id.
        new_date: the new date, YYYY-MM-DD - ideally a date you've confirmed
            (via get_forecast/get_air_quality/check_reschedule_needed) has
            better conditions.
        reason: the grounded reason for the move, from check_reschedule_needed.

    Returns:
        On success: the updated itinerary item dict (status='rescheduled').
        On failure: dict with an "error" key.
    """
    try:
        return store.reschedule_itinerary_item(item_id, new_date, reason)
    except store.TripStoreError as exc:
        return _err(exc)


# ---------------------------------------------------------------------------
# Packing list
# ---------------------------------------------------------------------------
@mcp.tool
def get_packing_list(trip_id: int) -> dict[str, Any]:
    """Get the current packing list for a trip.

    Args:
        trip_id: the trip id.

    Returns:
        On success: dict with an "items" list (item_name, category, quantity,
        reason, packed). On failure: dict with an "error" key.
    """
    try:
        return {"items": store.get_packing_list(trip_id)}
    except store.TripStoreError as exc:
        return _err(exc)


@mcp.tool
def add_packing_item(trip_id: int, item_name: str, category: str | None = None, quantity: int = 1, reason: str | None = None) -> dict[str, Any]:
    """Add an item to the packing list. Always include a specific `reason`
    tied to real forecast/AQI numbers when the item is weather-driven (e.g.
    "N95 mask - AQI forecast 165 on Aug 14 due to smoke") rather than a
    generic reason - pull the numbers from get_forecast/get_air_quality first.

    Args:
        trip_id: the trip id.
        item_name: e.g. "Compact umbrella", "N95 mask", "Hiking boots".
        category: e.g. "clothing", "gear", "documents", "weather", "health".
        quantity: how many (default 1).
        reason: why this item was added - be specific with dates/numbers when weather-driven.

    Returns:
        On success: the new packing item dict. On failure: dict with an "error" key.
    """
    try:
        return store.add_packing_item(trip_id, item_name, category, quantity, reason)
    except store.TripStoreError as exc:
        return _err(exc)


@mcp.tool
def remove_packing_item(item_id: int) -> dict[str, Any]:
    """Remove an item from the packing list.

    Args:
        item_id: the packing item id.

    Returns:
        On success: {"removed": True}. On failure: dict with an "error" key.
    """
    try:
        store.remove_packing_item(item_id)
        return {"removed": True}
    except store.TripStoreError as exc:
        return _err(exc)


@mcp.tool
def set_packing_item_packed(item_id: int, packed: bool) -> dict[str, Any]:
    """Mark a packing item as packed or not packed.

    Args:
        item_id: the packing item id.
        packed: True if packed, False if not.

    Returns:
        On success: the updated packing item dict. On failure: dict with an "error" key.
    """
    try:
        return store.set_packed(item_id, packed)
    except store.TripStoreError as exc:
        return _err(exc)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    mcp.run(transport="streamable-http", host="0.0.0.0", port=port)
