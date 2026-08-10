"""
embeddings.py - duplicated into pipeline/, mcp_server/, and frontend/ (see common/ for the reference copy & duplication note)

Turns free text (destination descriptions, attraction blurbs, activity
descriptions, user preference notes) into vectors for semantic retrieval,
and provides a cosine-similarity search over stored embeddings. This is
the "context engineering" / unstructured-data-processing piece of the
capstone requirements.

Two embedding backends:

1. Databricks Foundation Model embeddings (primary, used whenever Databricks
   credentials are available - i.e. when running as a Databricks App or Job).
   Calls a Databricks Model Serving embedding endpoint (default:
   "databricks-gte-large-en", a pay-per-token foundation model endpoint
   available by default in most Databricks workspaces) via the Databricks
   SDK. No separate API key needed - it uses the app/job's own Databricks
   auth.

2. Local fallback (hashing-trick vectorizer, scikit-learn's
   HashingVectorizer) - used automatically when Databricks credentials
   aren't available, e.g. running `pipeline/ingest.py` on a laptop without
   `databricks configure` set up, or in a CI/test environment. This keeps
   the whole pipeline runnable end-to-end with zero credentials, at the
   cost of a much weaker, purely lexical (not semantic) similarity signal.
   Swap to the Databricks backend for real semantic retrieval.

Embeddings are stored as JSON-encoded float lists (see db.py) so similarity
search happens in Python via cosine similarity, not in the database. This
keeps the code portable across SQLite (local dev) and Postgres/Lakebase
(production) without depending on the pgvector extension being enabled.
"""

from __future__ import annotations

import json
import math
import os

EMBEDDING_ENDPOINT = os.environ.get("EMBEDDING_ENDPOINT", "databricks-gte-large-en")
_LOCAL_EMBEDDING_DIM = 384

_hashing_vectorizer = None  # lazily constructed, local-fallback path only


def _local_fallback_embed(texts: list[str]) -> list[list[float]]:
    """Zero-dependency-beyond-scikit-learn fallback: a hashing-trick bag-of-words
    vector. Deterministic, no network/credentials required, but lexical
    (keyword-overlap) rather than truly semantic similarity."""
    global _hashing_vectorizer
    from sklearn.feature_extraction.text import HashingVectorizer

    if _hashing_vectorizer is None:
        _hashing_vectorizer = HashingVectorizer(n_features=_LOCAL_EMBEDDING_DIM, alternate_sign=False, norm="l2")
    matrix = _hashing_vectorizer.transform(texts)
    return [row.toarray()[0].tolist() for row in matrix]


def _databricks_embed(texts: list[str]) -> list[list[float]]:
    """Call a Databricks Model Serving embedding endpoint via the Databricks SDK.
    Raises if Databricks credentials aren't configured - callers should catch
    and fall back to _local_fallback_embed."""
    from databricks.sdk import WorkspaceClient

    client = WorkspaceClient()
    response = client.serving_endpoints.query(name=EMBEDDING_ENDPOINT, input=texts)
    return [item.embedding for item in response.data]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of strings. Tries the Databricks foundation model endpoint
    first, falls back to a local hashing vectorizer if Databricks credentials
    aren't available. Empty/whitespace-only strings get a zero vector."""
    cleaned = [t.strip() if t and t.strip() else "" for t in texts]
    non_empty_idx = [i for i, t in enumerate(cleaned) if t]
    if not non_empty_idx:
        return [[0.0] * _LOCAL_EMBEDDING_DIM for _ in texts]

    non_empty_texts = [cleaned[i] for i in non_empty_idx]
    try:
        vectors = _databricks_embed(non_empty_texts)
    except Exception:
        vectors = _local_fallback_embed(non_empty_texts)

    dim = len(vectors[0]) if vectors else _LOCAL_EMBEDDING_DIM
    result = [[0.0] * dim for _ in texts]
    for idx, vec in zip(non_empty_idx, vectors):
        result[idx] = vec
    return result


def embed_text(text: str) -> list[float]:
    """Embed a single string. Convenience wrapper around embed_texts."""
    return embed_texts([text])[0]


def to_json(vector: list[float]) -> str:
    """Serialize an embedding for storage in a TEXT column."""
    return json.dumps(vector)


def from_json(blob: str | None) -> list[float]:
    """Deserialize an embedding from a TEXT column. Returns [] for NULL/empty."""
    if not blob:
        return []
    try:
        return json.loads(blob)
    except (ValueError, TypeError):
        return []


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Standard cosine similarity, 0.0 if either vector is empty/zero or the
    dimensions don't match (e.g. mixing Databricks and local-fallback
    embeddings - don't do that in production, but don't crash on it either)."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def rank_by_similarity(query_embedding: list[float], candidates: list[dict], embedding_key: str = "embedding_json", top_k: int = 5) -> list[dict]:
    """Given a list of candidate dicts each holding a JSON-encoded embedding
    under `embedding_key`, return the top_k most similar, each with a
    `similarity` field added (0.0-1.0, higher is more similar)."""
    scored = []
    for candidate in candidates:
        vec = from_json(candidate.get(embedding_key))
        score = cosine_similarity(query_embedding, vec)
        scored.append({**candidate, "similarity": round(score, 4)})
    scored.sort(key=lambda c: c["similarity"], reverse=True)
    return scored[:top_k]
