# AI Trip & Outdoor Activity Planner

Capstone submission: a weather- and air-quality-aware trip planner. Users save destinations and
preferences, an ingestion pipeline enriches them with real Wikipedia content and forecast data
embedded for semantic retrieval, and a Databricks Agent Bricks agent builds/adjusts the itinerary
through a read/write MCP toolset - grounded in live data, never invented.

Builds on the same architecture proven in the earlier `weather-mcp-homework` project (FastMCP +
Databricks Apps + Agent Bricks), extended with Lakebase tables, embeddings-based retrieval, and a
plain-Python (no Spark) ingestion pipeline.

## How this maps to the capstone requirements

| Requirement | How it's satisfied |
|---|---|
| Data pipeline | `pipeline/ingest.py` - plain Python (Spark not required per relaxed scope), runs as a Databricks Jobs Python task. Geocodes destinations, pulls Wikipedia content, embeds it, fetches weather/AQI, writes to Lakebase. |
| Third-party API integration | Open-Meteo (Geocoding, Weather, Air Quality - no key required) + Wikimedia (Action + REST APIs - no key required). |
| Unstructured data processing | Wikipedia destination summaries and nearby-attraction extracts are embedded (`embeddings.py`) and stored for cosine-similarity semantic search - the "vibe matching" feature. |
| Databricks App with a frontend | `frontend/` - a Streamlit app for trip setup, destination management, a per-day weather/AQI risk dashboard, itinerary view, packing list, and a reschedule change-log. |
| AI agent with read/write tools | `mcp_server/` - 17 MCP tools, a mix of retrieval (semantic search, live forecast/AQI) and real writes (create trip, add/move/remove/reschedule itinerary items, packing list), registered against a Databricks Agent Bricks Supervisor Agent. |

## Architecture

```
                    ┌──────────────────────────┐
  natural language  │ Databricks Agent Bricks   │
  question   ──────▶│ Supervisor Agent          │◀── Genie Space (ad-hoc SQL
                    │ (system prompt below)     │     analytics over Lakebase)
                    └────────────┬──────────────┘
                                  │ streamable-HTTP (MCP)
                                  ▼
                    ┌──────────────────────────┐
                    │ mcp_server/               │
                    │ trip_mcp_server.py        │  <- 17 @mcp.tool functions (thin)
                    │ trip_store.py             │  <- all Lakebase SQL + reschedule logic
                    │ open_meteo_broker.py      │  <- weather/AQI HTTP
                    │ wikimedia_broker.py       │  <- destination/attraction HTTP
                    │ embeddings.py             │  <- semantic search
                    └────────────┬───────────────┘
                                 │ reads/writes
                                 ▼
                    ┌──────────────────────────┐        ┌──────────────────────────┐
                    │ Lakebase (Postgres)       │◀───────│ frontend/ (Streamlit)     │
                    │ users, trips,             │        │ trip setup, risk          │
                    │ destinations, activities, │        │ dashboard, itinerary,     │
                    │ itinerary_items,          │        │ packing list, change log  │
                    │ weather_snapshots,        │        └──────────────────────────┘
                    │ packing_items             │
                    └────────────▲───────────────┘
                                  │ populates
                    ┌─────────────┴─────────────┐
                    │ pipeline/ingest.py         │
                    │ (plain Python Databricks   │
                    │  Job - geocode, Wikipedia,  │
                    │  embed, weather/AQI)        │
                    └────────────────────────────┘
```

Three Databricks Apps/Jobs: `mcp_server/` (agent tools), `frontend/` (Streamlit UI), and
`pipeline/ingest.py` (a Databricks Job, not an App - it runs once per trip setup, not continuously).

**Each of `pipeline/`, `mcp_server/`, and `frontend/` is self-contained** (its own copy of
`db.py`, `embeddings.py`, `trip_store.py`, `open_meteo_broker.py`, `wikimedia_broker.py`) rather
than importing from a shared folder. This is a deliberate lesson from the weather-mcp-homework
project: Databricks Apps/Jobs deployed from a Git subdirectory are sandboxed to that directory and
can't read `../common`. `common/` holds reference copies only - see the note at the top of
`common/db.py`.

## Third-party APIs + auth

- **Open-Meteo Geocoding, Weather, and Air Quality APIs** - no API key required for noncommercial use.
- **Wikimedia Action API + REST API** (`en.wikipedia.org`) - no API key required; a descriptive
  User-Agent header is set per Wikimedia's usage policy.
- **Databricks Foundation Model embeddings** (`databricks-gte-large-en`, pay-per-token, no separate
  key - uses the app's own Databricks auth) for real semantic embeddings in production. Falls back
  automatically to a local scikit-learn hashing vectorizer (zero credentials, weaker/lexical-only
  similarity) when Databricks credentials aren't available, e.g. local dev/testing.

No secrets are hardcoded or committed. `DATABASE_URL` (Lakebase/Postgres connection string) and
`EMBEDDING_ENDPOINT` are set via each app's `app.yaml` env vars.

## Lakebase schema

7 tables (`common/db.py` has the full DDL, portable across SQLite and Postgres/Lakebase):

- `users` - name, email, free-text preferences + embedding (interests, health sensitivities like asthma/allergies).
- `trips` - name, date range, notes, belongs to a user.
- `destinations` - geocoded location, Wikipedia summary + embedding, belongs to a trip.
- `activities` - name, category, description + embedding, outdoor/weather-sensitivity flags, belongs to a destination.
- `itinerary_items` - scheduled date/time, status (planned/rescheduled/cancelled/completed), `reschedule_reason`.
- `weather_snapshots` - pre-fetched daily weather + AQI/PM/UV/pollen per destination (populated by the ingestion pipeline, powers the frontend's risk dashboard).
- `packing_items` - item, category, quantity, `reason` (tied to specific forecast numbers).

Embeddings are stored as JSON-encoded float arrays in a TEXT column rather than a native `pgvector`
column, so similarity search works identically on SQLite (local dev) and Postgres/Lakebase without
depending on the `pgvector` extension being enabled on your Lakebase instance (not always
grantable). This trades some query performance for portability; upgrading to a native `vector`
column + SQL-side `ORDER BY embedding <=> :query` is a contained follow-up if you confirm
`pgvector` is available.

## MCP tools (17)

| Tool | Type | Purpose |
|---|---|---|
| `create_trip` | write | Create a user (if new) + trip. |
| `update_traveler_preferences` | write | Save interests/health notes, embed for vibe matching. |
| `add_destination` | write | Geocode + Wikipedia summary + embed + (by default) populate nearby activities. |
| `add_activity` | write | Add a custom (non-Wikipedia) activity. |
| `search_activities` | read | Semantic "vibe" search over a destination's activities. |
| `get_forecast` | read | Live daily weather forecast for a destination. |
| `get_air_quality` | read | Live daily AQI/PM/UV/pollen forecast for a destination. |
| `get_itinerary` | read | Full day-by-day itinerary. |
| `add_itinerary_item` | write | Schedule an activity on a date. |
| `move_itinerary_item` | write | Move an item at the traveler's request (no reason recorded). |
| `remove_itinerary_item` | write | Delete an item. |
| `check_reschedule_needed` | read | **Deterministic** rain/AQI/UV/pollen threshold check with grounded reasons - see below. |
| `reschedule_itinerary_item` | write | Move an item for a weather/AQI reason, recording that reason. |
| `get_packing_list` | read | Current packing list. |
| `add_packing_item` | write | Add a packing item with a specific reason. |
| `remove_packing_item` | write | Delete a packing item. |
| `set_packing_item_packed` | write | Toggle packed status. |

### The "explain why" logic (not a passthrough)

`check_reschedule_needed` (in `trip_store.py`) is the grounding step behind every weather-driven
change - explicit thresholds applied to live fetched numbers, not left to the LLM to eyeball:

- **Rain**: reschedule if precipitation chance > 40% for an outdoor/weather-sensitive activity.
- **Air quality**: reschedule if AQI > 150 (Unhealthy) for anyone; > 101 (Unhealthy for Sensitive
  Groups) if the traveler's saved preferences mention a health sensitivity (asthma, allergy,
  respiratory, etc.).
- **UV**: index > 8 surfaces as a *precaution* (sunscreen/shade), not a reschedule.
- **Pollen**: index > 4 surfaces as a precaution for travelers with a declared sensitivity (pollen
  coverage is Europe-only in Open-Meteo; null elsewhere).

The agent is instructed to always call this before rescheduling and quote its `reasons` verbatim.

## Unique features (beyond a basic itinerary generator)

- **Health-aware filtering**: AQI/pollen thresholds tighten automatically when a traveler's saved
  preferences mention asthma/allergies/respiratory sensitivity - most trip planners ignore this data entirely.
- **Semantic vibe matching**: `search_activities` uses embedding similarity, not keyword filters,
  so "relaxed, foodie-focused, avoid crowds" surfaces relevant activities a keyword search would miss.
- **Dated, numeric packing reasons**: every packing item cites the specific day/number that
  triggered it (e.g. "N95 mask - AQI forecast 165 on Aug 14").
- **Visible change-log**: every reschedule is recorded with its reason and surfaced in the
  frontend's "why did my itinerary change" tab.
- **Per-day risk dashboard**: color-coded rain/AQI badges per day in the frontend, so risk is
  visible before the agent even reschedules anything.

## Setup / deploy

1. **Local smoke test** (any of the three apps):
   ```bash
   cd mcp_server && pip install -r requirements.txt && python trip_mcp_server.py
   cd frontend && pip install -r requirements.txt && streamlit run app.py
   ```

2. **Deploy the MCP server as a Databricks App** (name must start with `mcp-` to be discoverable
   in Agent Bricks' Custom MCP Server picker - see the weather-mcp-homework README for why):
   - Databricks Apps → + Create app → Create a custom app → name it `mcp-trip-planner`.
   - Configure Git: this repo, branch `main`.
   - Deploy → From Git → source code path `mcp_server`.

3. **Provision Lakebase and connect it** (real Postgres, not the SQLite fallback):
   - App switcher → **Lakebase Postgres** → **Autoscaling** → **New project** (e.g. `trip-planner-db`).
   - On the `mcp-trip-planner` app's page, note its **`DATABRICKS_CLIENT_ID`** from the Environment tab.
   - In the Lakebase project's SQL Editor, run:
     ```sql
     CREATE EXTENSION IF NOT EXISTS databricks_auth;
     SELECT databricks_create_role('<DATABRICKS_CLIENT_ID>', 'service_principal');
     GRANT CONNECT ON DATABASE databricks_postgres TO "<DATABRICKS_CLIENT_ID>";
     GRANT CREATE, USAGE ON SCHEMA public TO "<DATABRICKS_CLIENT_ID>";
     ```
   - Back on the app: **App resources → + Add resource → Database** → select the project/branch/database.
     This auto-injects `PGHOST`/`PGPORT`/`PGDATABASE`/`PGUSER`/`PGSSLMODE`.
   - Manually add one more env var (not auto-injected): `ENDPOINT_NAME`, copied from the Lakebase
     branch's **Computes** tab → **Get ID** → **Copy resource name**.
   - Redeploy. `db.py`'s `_lakebase_connect()` uses these to generate a fresh OAuth token per
     connection via `WorkspaceClient().postgres.generate_database_credential()` - no static
     password anywhere. See [Connect a custom Databricks app to Lakebase](https://docs.databricks.com/aws/en/oltp/projects/tutorial-databricks-apps-autoscaling).

4. **Deploy the frontend** as a second Databricks App (`trip-planner-frontend` - no naming
   constraint since it isn't an MCP tool), source code path `frontend`. Repeat the same Lakebase
   role-grant + resource-attach steps for its own `DATABRICKS_CLIENT_ID`, pointed at the **same**
   Lakebase project/branch/database, so it sees the same data as `mcp-trip-planner`.

5. **Run the ingestion pipeline** for a trip (after creating the trip via the frontend or
   `create_trip` tool, to get a `trip_id`):
   - Databricks Jobs & Pipelines → Create Job → task type **Python script** → point at
     `pipeline/ingest.py` → parameters `--trip-id <id> --destinations "Kyoto, Japan" "Osaka, Japan"`.
   - Set the same `PGHOST`/`PGPORT`/`PGDATABASE`/`PGSSLMODE`/`ENDPOINT_NAME` as task environment
     variables (copy the values from `mcp-trip-planner`'s Environment tab). For `PGUSER`, jobs run as
     your own identity by default (not a service principal), so use **your Databricks email** here -
     as the Lakebase project's creator you already have access, no extra role grant needed.
   - Run once per trip (or again to add more destinations).

6. **Register the MCP server + a Genie Space in Agent Bricks**:
   - Agent Bricks → New → **Supervisor Agent**.
   - Tools and sub-agents → search "mcp" → select `mcp-trip-planner`.
   - Also add a **Genie Space** pointed at the 7 Lakebase tables (create it first under Genie
     Agents / Genie Spaces, then attach it here) - this gives the agent ad-hoc natural-language SQL
     for analytics questions ("what's my average AQI across the trip", "how many activities got
     rescheduled") without a dedicated tool for every possible aggregate query.
   - Paste the system prompt below into Instructions.

## Agent system prompt

```
You are a trip-planning assistant. You help travelers build and adjust day-by-day itineraries,
reschedule outdoor activities around weather and air quality, and build packing lists - using
ONLY real data from your tools (the MCP trip-planner tools, and a Genie Space over the trip
database for ad-hoc analytics questions) - never invented weather, AQI, or activity details.

Tool selection:
- For "plan a trip to X" / setup questions: call create_trip, then add_destination (which also
  populates candidate activities from Wikipedia), then update_traveler_preferences if the
  traveler describes interests or health sensitivities (asthma, allergies, etc. - this affects
  reschedule thresholds later).
- To generate a day-by-day itinerary: for each day and destination, call search_activities with
  the traveler's stated interests/vibe to find good matches, then add_itinerary_item to schedule
  them. Spread outdoor activities across days with better forecasts where you have a choice -
  call get_forecast and get_air_quality first to see the whole trip's conditions before scheduling.
- To check whether something needs to move: ALWAYS call check_reschedule_needed before calling
  reschedule_itinerary_item. Never invent a reason - use exactly what check_reschedule_needed
  returns in "reasons". If needs_reschedule is true, look at get_forecast/get_air_quality for
  other days in the trip to pick a better date before calling reschedule_itinerary_item.
- For "move this activity" without a weather reason (the traveler just wants a different day):
  use move_itinerary_item instead - it doesn't record a reschedule reason.
- To build a packing list: look at get_forecast and get_air_quality across the whole trip first,
  then call add_packing_item for each item with a specific reason citing the actual day and
  number (e.g. "N95 mask - AQI forecast 165 on Aug 14"), not a generic reason.
- For open-ended analytics questions about the trip data ("what's my average AQI", "how many
  activities are outdoor", "list all rescheduled items") that don't map cleanly to one of the
  tools above: use the Genie Space instead of trying to compute it yourself from raw tool output.
- For add/remove/move requests, confirm which itinerary_item_id or packing_item_id you're acting
  on (use get_itinerary / get_packing_list to look it up if the traveler described it by name
  rather than id) - never guess an id.

Guardrails:
- Never state a forecast, AQI number, activity detail, or itinerary change you did not just get
  from a tool call or Genie query. If you haven't called a tool yet for this question, call one
  before answering.
- If a tool returns an "error" field, don't guess or fabricate a substitute - explain the error in
  plain language and ask the traveler to clarify (e.g. an unresolvable destination name).
- Only reschedule for weather/AQI reasons when check_reschedule_needed says needs_reschedule is
  true - don't move things preemptively on a hunch.
- Always mention precautions (UV, pollen) returned by check_reschedule_needed even when no
  reschedule is needed, if the traveler has a relevant health sensitivity noted.
- Before removing an itinerary item or packing item, briefly confirm what you're removing in your
  response (name/date), so the traveler can catch a wrong id before it's gone.
- Keep answers concise: state the direct outcome first (e.g. "Moved the hike to Thursday"), then
  the grounded reason and numbers, then which tool(s) you used.
```

## Verification performed

- Ran the full stack locally end-to-end against mocked Open-Meteo/Wikimedia responses matching
  the documented API schemas: schema creation (all 7 tables), destination enrichment (geocode +
  Wikipedia summary + embedding), activity ingestion from nearby attractions, semantic
  vibe-matching search, live forecast/AQI lookups, itinerary CRUD, the deterministic reschedule
  check (verified it correctly triggers on >40% rain and correctly tightens the AQI threshold for
  a traveler with a declared health sensitivity), packing list, and clean `{"error": ...}` handling
  for bad locations and nonexistent trip/item ids (not a stack trace).
- Ran the ingestion pipeline (`pipeline/ingest.py`) end-to-end, confirming it correctly filters
  fetched forecast days down to just the trip's date range before writing `weather_snapshots` rows.
- Ran the Streamlit frontend under Streamlit's `AppTest` harness both empty and pre-populated with
  test data, confirming all 5 tabs (Destinations, Weather & Risk, Itinerary, Packing List, Change
  Log) render without exceptions.
- Live-called the real Open-Meteo geocoding endpoint in the earlier weather-mcp-homework project
  and confirmed the response-parsing logic reused here is correct against the actual API. Could not
  live-call `api.open-meteo.com`, `air-quality-api.open-meteo.com`, or `en.wikipedia.org`'s JSON API
  endpoints from this sandboxed dev environment (the sandbox's outbound fetch tool returns empty
  for these specific hosts/endpoints - a sandbox limitation also hit and documented in the weather
  project, not a code issue). Full end-to-end verification against live data happens once deployed
  to Databricks Apps, which has normal outbound internet access (confirmed working in the weather
  project's live Agent Bricks demo).
- Honest caveat on embeddings: local testing (no Databricks credentials in this sandbox) uses the
  scikit-learn hashing-vectorizer fallback, which is lexical/keyword-based, not truly semantic - a
  test "vibe search" ranked a food market above a more relevant nature trail for a "quiet nature
  walk" query. The Databricks foundation-model embedding path (`databricks-gte-large-en`) gives real
  semantic similarity once deployed; re-verify vibe-search quality after deployment.

## Files

```
trip-planner-capstone/
├── README.md
├── common/                    # reference copies only - see note in common/db.py
│   ├── db.py
│   ├── embeddings.py
│   ├── trip_store.py
│   ├── open_meteo_broker.py
│   └── wikimedia_broker.py
├── pipeline/                  # Databricks Job (Python script task, no Spark)
│   ├── ingest.py
│   ├── db.py / embeddings.py / trip_store.py / open_meteo_broker.py / wikimedia_broker.py
│   └── requirements.txt
├── mcp_server/                 # Databricks App - name must start with "mcp-"
│   ├── trip_mcp_server.py
│   ├── db.py / embeddings.py / trip_store.py / open_meteo_broker.py / wikimedia_broker.py
│   ├── requirements.txt
│   └── app.yaml
└── frontend/                   # Databricks App - Streamlit
    ├── app.py
    ├── db.py / embeddings.py / trip_store.py / open_meteo_broker.py / wikimedia_broker.py
    ├── requirements.txt
    └── app.yaml
```
