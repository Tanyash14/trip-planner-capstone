"""
wikimedia_broker.py - duplicated into pipeline/, mcp_server/, and frontend/ (see common/ for the reference copy & duplication note)

Adapter module for the Wikimedia/Wikipedia APIs: destination descriptions
and nearby points of interest ("attractions"), used to seed the
`destinations` and `activities` tables with real unstructured text that
then gets embedded for semantic retrieval (see embeddings.py).

Uses the public English Wikipedia REST + Action APIs. No API key required;
Wikimedia asks for a descriptive User-Agent identifying the app (see
https://meta.wikimedia.org/wiki/User-Agent_policy), which is set below.
"""

from __future__ import annotations

from typing import Any

import requests

WIKI_API_URL = "https://en.wikipedia.org/w/api.php"
WIKI_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
REQUEST_TIMEOUT_SECS = 10
USER_AGENT = "trip-planner-capstone/1.0 (educational project; contact: shanya1496@gmail.com)"


class WikimediaLookupError(Exception):
    """Raised for any recoverable failure: no matching article, API outage.
    Callers catch this and return a clean {"error": ...} dict rather than
    letting a missing Wikipedia article break the whole pipeline."""


def _get(url: str, params: dict[str, Any] | None = None) -> dict:
    try:
        resp = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT_SECS)
    except requests.exceptions.Timeout as exc:
        raise WikimediaLookupError(f"Wikimedia timed out calling {url}") from exc
    except requests.exceptions.RequestException as exc:
        raise WikimediaLookupError(f"Could not reach Wikimedia ({url}): {exc}") from exc

    if resp.status_code == 404:
        raise WikimediaLookupError(f"No Wikipedia article found ({url})")
    if not resp.ok:
        raise WikimediaLookupError(f"Wikimedia returned an error ({resp.status_code}) for {url}")
    try:
        return resp.json()
    except ValueError as exc:
        raise WikimediaLookupError(f"Wikimedia returned an unreadable response from {url}") from exc


def search_title(query: str) -> str | None:
    """Find the best-matching Wikipedia article title for a free-text query
    (e.g. a destination name). Returns None if nothing matches, rather than
    raising - a missing article isn't fatal for the pipeline."""
    query = (query or "").strip()
    if not query:
        return None
    data = _get(WIKI_API_URL, params={
        "action": "query", "list": "search", "srsearch": query,
        "srlimit": 1, "format": "json",
    })
    results = data.get("query", {}).get("search", [])
    return results[0]["title"] if results else None


def get_destination_summary(location_name: str) -> dict[str, Any]:
    """Look up a Wikipedia summary for a destination.

    Args:
        location_name: free-text destination name, e.g. "Kyoto, Japan".

    Returns:
        dict with title, extract (description text), url, thumbnail_url
        (may be None). If no article is found, extract/title are None
        rather than raising - some destinations genuinely have no article.
    """
    title = search_title(location_name)
    if not title:
        return {"title": None, "extract": None, "url": None, "thumbnail_url": None}

    data = _get(WIKI_SUMMARY_URL.format(title=title.replace(" ", "_")))
    return {
        "title": data.get("title", title),
        "extract": data.get("extract"),
        "url": (data.get("content_urls", {}).get("desktop", {}) or {}).get("page"),
        "thumbnail_url": (data.get("thumbnail") or {}).get("source"),
    }


def get_nearby_attractions(latitude: float, longitude: float, radius_m: int = 10000, limit: int = 15) -> list[dict[str, Any]]:
    """Find Wikipedia articles for points of interest near a coordinate
    (geosearch), each with a short plain-text extract - a good source of
    "activities" (landmarks, museums, parks, etc.) for a destination.

    Args:
        latitude, longitude: destination coordinates.
        radius_m: search radius in meters, max 10000 per the Wikimedia API.
        limit: max number of results, max 500 per the API (kept small here).

    Returns:
        list of dicts: title, extract, distance_m, url.
    """
    radius_m = min(radius_m, 10000)
    geo_data = _get(WIKI_API_URL, params={
        "action": "query", "list": "geosearch",
        "gscoord": f"{latitude}|{longitude}", "gsradius": radius_m,
        "gslimit": limit, "format": "json",
    })
    places = geo_data.get("query", {}).get("geosearch", [])
    if not places:
        return []

    pageids = [str(p["pageid"]) for p in places]
    extract_data = _get(WIKI_API_URL, params={
        "action": "query", "prop": "extracts|info",
        "exintro": 1, "explaintext": 1, "exsentences": 3,
        "inprop": "url", "pageids": "|".join(pageids), "format": "json",
    })
    pages = extract_data.get("query", {}).get("pages", {})
    dist_by_id = {str(p["pageid"]): p.get("dist") for p in places}

    out = []
    for pageid, page in pages.items():
        out.append({
            "title": page.get("title"),
            "extract": page.get("extract") or "",
            "distance_m": dist_by_id.get(pageid),
            "url": page.get("fullurl"),
        })
    out.sort(key=lambda p: (p["distance_m"] is None, p["distance_m"]))
    return out
