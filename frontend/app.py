"""
app.py

Streamlit frontend for the trip planner - the capstone's "Databricks App
with a frontend" requirement. This is a data-management/visualization
cockpit (create trips, add destinations, view the itinerary, packing list,
per-day weather/AQI risk dashboard, and a change-log of agent-made
reschedules) that reads/writes the same Lakebase tables the MCP server's
agent tools use. The actual conversational AI agent lives in Databricks
Agent Bricks, registered against `mcp_server/` (see the weather-mcp-homework
project's pattern) - this app is the human-facing companion to it, not a
replacement for it.

Run locally:
    streamlit run app.py
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

import db
import open_meteo_broker as weather
import trip_store as store

st.set_page_config(page_title="AI Trip & Outdoor Activity Planner", page_icon="\U0001F9ED", layout="wide")
db.init_schema()

RISK_COLORS = {"low": "🟢", "moderate": "🟡", "high": "🔴"}


def rain_risk(pct):
    if pct is None:
        return "low"
    return "high" if pct > 40 else ("moderate" if pct >= 20 else "low")


def aqi_risk(aqi):
    if aqi is None:
        return "low"
    return "high" if aqi > 150 else ("moderate" if aqi > 101 else "low")


# ---------------------------------------------------------------------------
# Sidebar: user + trip selection / creation (Layla-style progressive setup)
# ---------------------------------------------------------------------------
st.sidebar.title("🧭 Trip Planner")

with st.sidebar.expander("1. Who's traveling?", expanded=True):
    user_name = st.text_input("Your name", key="user_name")
    user_email = st.text_input("Email (optional, used to find your trips again)", key="user_email")
    preferences_text = st.text_area(
        "Interests & notes (used for activity matching)",
        placeholder="e.g. loves hiking and quiet nature spots, not into big crowds, has asthma so sensitive to air quality",
        key="preferences_text",
    )
    if st.button("Save traveler profile"):
        if user_name:
            user = store.get_or_create_user(user_name, user_email or None)
            if preferences_text:
                store.update_user_preferences(user["id"], preferences_text)
            st.session_state["user_id"] = user["id"]
            st.success(f"Saved - user_id {user['id']}")
        else:
            st.warning("Enter a name first.")

user_id = st.session_state.get("user_id")

if user_id:
    with st.sidebar.expander("2. Trip details", expanded=True):
        existing_trips = store.list_trips(user_id)
        trip_names = ["+ New trip"] + [f"{t['name']} ({t['id']})" for t in existing_trips]
        choice = st.selectbox("Trip", trip_names)

        if choice == "+ New trip":
            trip_name = st.text_input("Trip name", placeholder="Japan Trip 2026")
            col1, col2 = st.columns(2)
            start_date = col1.date_input("Start date", value=dt.date.today())
            end_date = col2.date_input("End date", value=dt.date.today() + dt.timedelta(days=6))
            notes = st.text_area("Trip notes (optional)")
            if st.button("Create trip"):
                if trip_name:
                    trip = store.create_trip(user_id, trip_name, start_date.isoformat(), end_date.isoformat(), notes or None)
                    st.session_state["trip_id"] = trip["id"]
                    st.success(f"Created trip {trip['id']}")
                    st.rerun()
                else:
                    st.warning("Enter a trip name.")
        else:
            trip_id = int(choice.split("(")[-1].rstrip(")"))
            st.session_state["trip_id"] = trip_id

trip_id = st.session_state.get("trip_id")

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
if not user_id or not trip_id:
    st.title("🧭 AI Trip & Outdoor Activity Planner")
    st.markdown(
        "Save a traveler profile and create (or pick) a trip in the sidebar to get started.\n\n"
        "Once a trip exists, chat with the trip-planning agent in Databricks Agent Bricks "
        "(registered against this project's MCP server) to generate an itinerary, reschedule "
        "activities around weather/air quality, and build a packing list - the results show up "
        "here automatically."
    )
    st.stop()

trip = store.get_trip(trip_id)
st.title(f"🧭 {trip['name']}")
st.caption(f"{trip['start_date']} to {trip['end_date']}" + (f" — {trip['notes']}" if trip.get("notes") else ""))

tab_dest, tab_risk, tab_itin, tab_pack, tab_log = st.tabs(
    ["📍 Destinations", "🌦️ Weather & Risk", "🗓️ Itinerary", "🎒 Packing List", "📝 Change Log"]
)

# --- Destinations tab ---
with tab_dest:
    st.subheader("Add a destination")
    st.caption("Geocodes the location, pulls a Wikipedia summary, and finds nearby attractions as candidate activities.")
    new_dest = st.text_input("Destination (city, country)", placeholder="Kyoto, Japan")
    if st.button("Add destination"):
        if new_dest:
            try:
                with st.spinner(f"Looking up {new_dest} and nearby attractions..."):
                    destination = store.add_destination(trip_id, new_dest)
                    activities = store.add_activities_from_attractions(destination["id"])
                st.success(f"Added {destination['resolved_name']} with {len(activities)} candidate activities.")
                st.rerun()
            except (store.TripStoreError, weather.WeatherLookupError) as exc:
                st.error(str(exc))

    destinations = store.list_destinations(trip_id)
    if not destinations:
        st.info("No destinations yet - add one above.")
    for d in destinations:
        with st.expander(f"📍 {d['resolved_name'] or d['name']}"):
            st.write(d.get("description") or "_No description available._")
            ph = "%s" if db.use_postgres() else "?"
            all_activities = db.run_query(f"SELECT * FROM activities WHERE destination_id = {ph}", (d["id"],))
            if all_activities:
                df = pd.DataFrame(all_activities)[["id", "name", "category", "is_outdoor", "requires_good_weather"]]
                st.dataframe(df, use_container_width=True, hide_index=True)

# --- Weather & Risk tab ---
with tab_risk:
    st.subheader("Per-day risk dashboard")
    st.caption("🟢 low risk · 🟡 moderate · 🔴 high — based on rain chance and air quality forecast, populated by the ingestion pipeline.")
    destinations = store.list_destinations(trip_id)
    if not destinations:
        st.info("Add a destination first.")
    for d in destinations:
        st.markdown(f"**{d['resolved_name'] or d['name']}**")
        snapshots = store.get_weather_snapshots(d["id"])
        if not snapshots:
            st.caption("No weather data yet - run the ingestion pipeline for this trip.")
            continue
        cols = st.columns(len(snapshots)) if len(snapshots) <= 8 else st.columns(8)
        for i, snap in enumerate(snapshots[:8]):
            with cols[i % len(cols)]:
                rr, ar = rain_risk(snap["precip_chance_pct"]), aqi_risk(snap["aqi"])
                st.markdown(f"**{snap['date'][5:]}**")
                st.markdown(f"{RISK_COLORS[rr]} Rain {snap['precip_chance_pct']}%")
                st.markdown(f"{RISK_COLORS[ar]} AQI {snap['aqi']}")
                st.caption(snap["conditions"] or "")

# --- Itinerary tab ---
with tab_itin:
    st.subheader("Day-by-day itinerary")
    items = store.get_itinerary(trip_id)
    if not items:
        st.info("No itinerary items yet - ask the trip-planning agent to generate one, or add items via chat.")
    else:
        df = pd.DataFrame(items)[["scheduled_date", "start_time", "activity_name", "destination_name", "status", "reschedule_reason"]]
        df = df.rename(columns={
            "scheduled_date": "Date", "start_time": "Time", "activity_name": "Activity",
            "destination_name": "Destination", "status": "Status", "reschedule_reason": "Reschedule reason",
        })
        st.dataframe(df, width="stretch", hide_index=True)

# --- Packing List tab ---
with tab_pack:
    st.subheader("Packing list")
    items = store.get_packing_list(trip_id)
    if not items:
        st.info("No packing items yet - ask the trip-planning agent to build one based on the forecast.")
    else:
        for item in items:
            col1, col2 = st.columns([1, 6])
            packed = col1.checkbox("Packed", value=bool(item["packed"]), key=f"pack_{item['id']}", label_visibility="collapsed")
            if packed != bool(item["packed"]):
                store.set_packed(item["id"], packed)
                st.rerun()
            label = f"**{item['item_name']}** ({item['category'] or 'general'}) x{item['quantity']}"
            if item.get("reason"):
                label += f"  \n_{item['reason']}_"
            col2.markdown(label)

# --- Change Log tab ---
with tab_log:
    st.subheader("Why did my itinerary change?")
    st.caption("Every weather/air-quality-driven reschedule the agent makes is logged here with its reason.")
    items = [i for i in store.get_itinerary(trip_id) if i["status"] == "rescheduled"]
    if not items:
        st.info("No reschedules yet.")
    for item in items:
        st.markdown(f"**{item['activity_name']}** moved to **{item['scheduled_date']}**")
        st.caption(item.get("reschedule_reason") or "No reason recorded.")
        st.divider()
