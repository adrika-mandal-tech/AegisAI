---
name: aegis-defensive-client-wrapper
description: When a service depends on an external resource like a database that might be unavailable, wrap the client library to handle connection errors gracefully, log warnings, and allow the main application to continue running in a degraded or no-op mode instead of crashing.
---

# Defensive Client Wrapper for External Services

## The Problem

When a core service (like the AegisAI backend) has a hard dependency on an external resource (like the Qdrant vector database), the entire service can crash if the external resource is unavailable. This leads to a brittle system and a poor user experience, as seen when the backend repeatedly failed due to Qdrant connection issues.

Simply retrying the connection or restarting the main service is often not enough and can lead to repetitive failure loops.

## The Solution

The robust solution is to make the client for the external service defensive. Instead of letting connection errors propagate and crash the application, the client wrapper should handle them gracefully.

This pattern involves:

1.  **Graceful Initialization:** In the client wrapper's `__init__`, wrap the connection attempt in a `try...except` block. If it fails, log a warning and set the internal client object to `None`. This allows the application to start up even if the dependency is down.

2.  **No-Op Methods:** In every method that interacts with the client object, first check if the client was initialized successfully. If not, return a sensible default value immediately (e.g., `None`, `False`, an empty list `[]`). This prevents `AttributeError` exceptions on the `None` client object and allows the application to run in a "degraded" or "no-op" mode.

3.  **Error Handling on Operations:** Even if the initial connection succeeded, subsequent operations can fail. Wrap all network calls within the client's methods in `try...except` blocks. Log any exceptions and return a default value, rather than letting the exception bubble up.

4.  **Boolean Status Returns:** For methods that are expected to perform a critical action (like `ensure_collection`), return a boolean `True`/`False` to indicate success or failure. This allows the calling code to know whether the operation worked and react accordingly, without needing to handle exceptions itself.

### Example: `QdrantDB` Wrapper

The fix applied to `backend/aegis/memory/qdrant.py` is a perfect example of this pattern.

-   The `__init__` method catches exceptions during client creation and sets `self._client = None`.
-   All public methods (`ensure_collection`, `upsert`, `search`, `delete_collection`) check `if not self._client:` at the beginning.
-   All external calls (`self._client.get_collections()`, `self._client.upsert()`, etc.) are wrapped in `try...except Exception`.
-   `ensure_collection` was changed to return `True` or `False`.
