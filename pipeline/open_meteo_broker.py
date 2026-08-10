"""
open_meteo_broker.py - duplicated into pipeline/, mcp_server/, and frontend/ (see common/ for the reference copy & duplication note)

Adapter module for Open-Meteo's Geocoding, Weather Forecast, and Air
Quality APIs. All HTTP calls and response parsing live here; callers (MCP
tools, the ingestion pipeline) get back clean dicts and never touch
`requests` directly. Same role as `weather_broker.py` in the earlier
weather-mcp-homework project, extended with hourly forecasts and air
quality/UV/pollen data for trip planning.

None of these Open-Meteo endpoints require an API key for noncommercial use
under the free tier limits.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import requests

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

REQUEST_TIMEOUT_SECS = 10

WMO_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "depositing rime fog",
    51: "light drizzle", 53: "moderate drizzle", 55: "dense drizzle",
    56: "light freezing drizzle", 57: "dense freezing drizzle",
    61: "slight rain", 63: "moderate rain", 65: "heavy rain",
    66: "light freezing rain", 67: "heavy freezing rain",
    71: "slight snow fall", 73: "moderate snow fall", 75: "heavy snow fall",
    77: "snow grains",
    80: "slight rain showers", 81: "moderate rain showers", 82: "violent rain showers",
    85: "slight snow showers", 86: "heavy snow showers",
    95: "thunderstorm", 96: "thunderstorm with slight hail", 99: "thunderstorm with heavy hail",
}


class WeatherLookupError(Exception):
    """Raised for any recoverable failure: bad location, API outage, bad params.
    Callers (MCP tools) catch this and return a clean {"error": ...} dict."""


def _get(url: str, params: dict[str, Any] | None = None) -> dict:
    try:
        resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_SECS)
    except requests.exceptions.Timeout as exc:
        raise WeatherLookupError(f"Weather service timed out calling {url}") from exc
    except requests.exceptions.RequestException as exc:
        raise WeatherLookupError(f"Could not reach weather service ({url}): {exc}") from exc

    if resp.status_code == 404:
        raise WeatherLookupError(f"No data found for this request at {url}")
    if not resp.ok:
        raise WeatherLookupError(f"Weather service returned an error ({resp.status_code}) for {url}")
    try:
        return resp.json()
    except ValueError as exc:
        raise WeatherLookupError(f"Weather service returned an unreadable response from {url}") from exc


def geocode_location(location: str) -> dict[str, Any]:
    """Resolve a free-text location to coordinates.

    Args:
        location: city name, "City, State/Country", or "lat,lon".

    Returns:
        dict with name, admin1, country, latitude, longitude, timezone.
    """
    location = (location or "").strip()
    if not location:
        raise WeatherLookupError("No location was provided.")

    if "," in location:
        parts = [p.strip() for p in location.split(",")]
        if len(parts) == 2:
            try:
                lat, lon = float(parts[0]), float(parts[1])
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    return {"name": location, "admin1": None, "country": None,
                            "latitude": lat, "longitude": lon, "timezone": "auto"}
            except ValueError:
                pass

    data = _get(GEOCODE_URL, params={"name": location, "count": 1, "language": "en", "format": "json"})
    results = data.get("results") or []
    if not results:
        raise WeatherLookupError(
            f"Could not find a location matching '{location}'. Try a city name, "
            f"'City, Country', or 'lat,lon'."
        )
    top = results[0]
    return {
        "name": top.get("name"), "admin1": top.get("admin1"), "country": top.get("country"),
        "latitude": top.get("latitude"), "longitude": top.get("longitude"),
        "timezone": top.get("timezone", "auto"),
    }


def _format_place(place: dict[str, Any]) -> str:
    parts = [p for p in (place.get("name"), place.get("admin1"), place.get("country")) if p]
    return ", ".join(parts) if parts else f"{place['latitude']},{place['longitude']}"


def get_daily_forecast(latitude: float, longitude: float, days: int, timezone: str = "auto") -> list[dict[str, Any]]:
    """Fetch a daily forecast (already-geocoded coordinates).

    Returns a list of per-day dicts: date, high_f, low_f, precip_chance_pct,
    precip_total_in, conditions.
    """
    days = max(1, min(int(days), 16))
    data = _get(FORECAST_URL, params={
        "latitude": latitude, "longitude": longitude,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,precipitation_sum,weather_code",
        "temperature_unit": "fahrenheit", "precipitation_unit": "inch",
        "timezone": timezone or "auto", "forecast_days": days,
    })
    daily = data.get("daily")
    if not daily or not daily.get("time"):
        raise WeatherLookupError(f"No forecast data returned for {latitude},{longitude}.")

    out = []
    for i, date_str in enumerate(daily["time"]):
        code = _safe_index(daily.get("weather_code"), i)
        out.append({
            "date": date_str,
            "high_f": _safe_index(daily.get("temperature_2m_max"), i),
            "low_f": _safe_index(daily.get("temperature_2m_min"), i),
            "precip_chance_pct": _safe_index(daily.get("precipitation_probability_max"), i),
            "precip_total_in": _safe_index(daily.get("precipitation_sum"), i),
            "conditions": WMO_CODES.get(code, f"unknown (code {code})" if code is not None else "unknown"),
        })
    return out


def get_hourly_forecast(latitude: float, longitude: float, days: int, timezone: str = "auto") -> list[dict[str, Any]]:
    """Fetch an hourly forecast (already-geocoded coordinates), useful for
    scheduling activities within a day.

    Returns a list of per-hour dicts: time (ISO), temp_f, precip_chance_pct, conditions.
    """
    days = max(1, min(int(days), 16))
    data = _get(FORECAST_URL, params={
        "latitude": latitude, "longitude": longitude,
        "hourly": "temperature_2m,precipitation_probability,weather_code",
        "temperature_unit": "fahrenheit", "timezone": timezone or "auto", "forecast_days": days,
    })
    hourly = data.get("hourly")
    if not hourly or not hourly.get("time"):
        raise WeatherLookupError(f"No hourly forecast data returned for {latitude},{longitude}.")

    out = []
    for i, time_str in enumerate(hourly["time"]):
        code = _safe_index(hourly.get("weather_code"), i)
        out.append({
            "time": time_str,
            "temp_f": _safe_index(hourly.get("temperature_2m"), i),
            "precip_chance_pct": _safe_index(hourly.get("precipitation_probability"), i),
            "conditions": WMO_CODES.get(code, f"unknown (code {code})" if code is not None else "unknown"),
        })
    return out


def get_air_quality(latitude: float, longitude: float, days: int, timezone: str = "auto") -> list[dict[str, Any]]:
    """Fetch daily-aggregated air quality (already-geocoded coordinates):
    US AQI, PM2.5, PM10, UV index, and pollen (where covered - pollen data
    is Europe-only in Open-Meteo; other regions get null pollen fields).

    The Air Quality API only returns hourly data, so this aggregates to one
    row per day (max for AQI/PM/UV - the worst-case value for the day,
    which is what matters for an outdoor-activity go/no-go decision).

    Returns a list of per-day dicts: date, aqi_max, pm2_5_max, pm10_max,
    uv_index_max, pollen_index_max (may be null).
    """
    days = max(1, min(int(days), 7))  # air quality forecast is capped shorter than weather
    hourly_fields = ["us_aqi", "pm2_5", "pm10", "uv_index"]
    pollen_fields = ["alder_pollen", "birch_pollen", "grass_pollen", "mugwort_pollen", "olive_pollen", "ragweed_pollen"]
    data = _get(AIR_QUALITY_URL, params={
        "latitude": latitude, "longitude": longitude,
        "hourly": ",".join(hourly_fields + pollen_fields),
        "timezone": timezone or "auto", "forecast_days": days,
    })
    hourly = data.get("hourly")
    if not hourly or not hourly.get("time"):
        raise WeatherLookupError(f"No air quality data returned for {latitude},{longitude}.")

    by_date: dict[str, dict[str, list[float]]] = {}
    for i, time_str in enumerate(hourly["time"]):
        date_str = time_str.split("T")[0]
        bucket = by_date.setdefault(date_str, {f: [] for f in hourly_fields + pollen_fields})
        for field in hourly_fields + pollen_fields:
            val = _safe_index(hourly.get(field), i)
            if val is not None:
                bucket[field].append(val)

    out = []
    for date_str in sorted(by_date.keys()):
        vals = by_date[date_str]
        pollen_values = [v for f in pollen_fields for v in vals[f]]
        out.append({
            "date": date_str,
            "aqi_max": max(vals["us_aqi"]) if vals["us_aqi"] else None,
            "pm2_5_max": round(max(vals["pm2_5"]), 1) if vals["pm2_5"] else None,
            "pm10_max": round(max(vals["pm10"]), 1) if vals["pm10"] else None,
            "uv_index_max": round(max(vals["uv_index"]), 1) if vals["uv_index"] else None,
            "pollen_index_max": round(max(pollen_values), 1) if pollen_values else None,
        })
    return out


def _safe_index(values: list | None, i: int):
    if not values or i >= len(values):
        return None
    return values[i]
