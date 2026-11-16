---
name: fastapi-in-memory-history-scoping
description: When adding in-memory request history to a FastAPI route, declare the mutable state as a module-level list, not inside the route function.
---

# FastAPI In-Memory History Scoping

A common pitfall when adding in-memory upload/request history to a FastAPI application: defining the history list *inside* the route function causes a fresh empty list on every request, so history is never persisted.

## The anti-pattern

```python
@router.post("/documents/upload")
async def upload_document(file: UploadFile):
    # BUG: This list is created anew on every request
    _history: list[dict] = []
    # ... process file, then try to append to _history ...
```

This defeats the purpose — every HTTP request creates a new function frame with a fresh `[]`.

## The fix

Move the history list to **module level**, so it lives for the lifetime of the process:

```python
# Near top of routes.py, after imports and constants
_history: list[dict] = []

@router.post("/documents/upload")
async def upload_document(file: UploadFile):
    # ... process file ...
    _history.append({
        "id": str(uuid.uuid4()),
        "filename": file.filename,
        "status": "success" if success else "failure",
        "error": error_message,
    })
```

## Why this works

- Module-level variables in Python are initialized once when the module is first imported.
- Uvicorn (via `--reload`) reloads the module on file changes, but within a single worker process the list persists across requests.
- For production, replace the in-memory list with a database-backed store (SQLite, PostgreSQL, etc.).

## Best practices

- Always use a `finally` block (or restructure to guarantee history recording):
  ```python
  try:
      # ... processing ...
  finally:
      _history.append(entry)  # Always recorded, even on exception
  ```
- Add status tracking variables (`status`, `error_message`) before the try block so they're in scope for both success and failure branches.
- Update the schema to include status fields so the read endpoint can return them.