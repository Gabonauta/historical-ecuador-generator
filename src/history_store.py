"""SQLite-backed persistence for generation history."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from src.utils import safe_str


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "app_state.sqlite3"

JSON_COLUMNS = {
    "retrieved_chunks_json",
    "entity_snapshot_json",
    "text_result_json",
    "image_result_json",
    "request_options_json",
}

BOOLEAN_COLUMNS = {
    "generate_text",
    "generate_image",
    "use_llm",
    "use_rag",
}

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS generation_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    entity_id TEXT,
    entity_name TEXT,
    entity_type TEXT,
    output_type TEXT,
    generate_text INTEGER NOT NULL DEFAULT 0,
    generate_image INTEGER NOT NULL DEFAULT 0,
    requested_llm_provider TEXT,
    effective_text_provider TEXT,
    text_mode TEXT,
    requested_image_provider TEXT,
    effective_image_provider TEXT,
    use_llm INTEGER NOT NULL DEFAULT 0,
    use_rag INTEGER NOT NULL DEFAULT 0,
    embedding_provider TEXT,
    image_mode TEXT,
    visual_style TEXT,
    image_size TEXT,
    text_error TEXT,
    image_error TEXT,
    generated_text TEXT,
    image_path TEXT,
    image_url TEXT,
    prompt_text TEXT,
    prompt_image TEXT,
    base_context_text TEXT,
    retrieved_context_text TEXT,
    retrieved_chunks_json TEXT,
    entity_snapshot_json TEXT NOT NULL,
    text_result_json TEXT,
    image_result_json TEXT,
    request_options_json TEXT NOT NULL
);
"""


def init_store(db_path: Path | None = None) -> None:
    """Create the local SQLite store when it does not already exist."""
    resolved_path = db_path or DEFAULT_DB_PATH
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    with _connect(resolved_path) as connection:
        connection.execute(CREATE_TABLE_SQL)
        connection.commit()


def save_generation_run(
    entity: dict[str, Any],
    request: dict[str, Any],
    result: dict[str, Any],
    db_path: Path | None = None,
) -> int:
    """Persist a complete generation run and return its database id."""
    resolved_path = db_path or DEFAULT_DB_PATH
    init_store(resolved_path)

    text_result = result.get("text_result") or {}
    image_result = result.get("image_result") or {}
    primary_context_owner = text_result or image_result
    primary_retrieved_owner = text_result or image_result

    payload = {
        "entity_id": safe_str(entity.get("id")) or safe_str(result.get("entity_id")),
        "entity_name": safe_str(entity.get("nombre")),
        "entity_type": safe_str(entity.get("tipo")),
        "output_type": safe_str(result.get("output_type")) or safe_str(request.get("output_type")),
        "generate_text": _bool_to_int(request.get("generate_text", bool(text_result))),
        "generate_image": _bool_to_int(request.get("generate_image", bool(image_result))),
        "requested_llm_provider": safe_str(request.get("llm_provider")) or None,
        "effective_text_provider": safe_str(text_result.get("provider")) or None,
        "text_mode": safe_str(text_result.get("mode")) or None,
        "requested_image_provider": safe_str(request.get("image_provider")) or None,
        "effective_image_provider": safe_str(image_result.get("provider")) or None,
        "use_llm": _bool_to_int(request.get("use_llm", False)),
        "use_rag": _bool_to_int(
            request.get(
                "use_rag",
                text_result.get("use_rag", image_result.get("use_rag", False)),
            )
        ),
        "embedding_provider": (
            safe_str(text_result.get("embedding_provider"))
            or safe_str(image_result.get("embedding_provider"))
            or safe_str(request.get("embedding_provider"))
            or None
        ),
        "image_mode": safe_str(image_result.get("image_mode")) or safe_str(request.get("image_mode")) or None,
        "visual_style": safe_str(image_result.get("visual_style")) or safe_str(request.get("visual_style")) or None,
        "image_size": safe_str(image_result.get("size")) or safe_str(request.get("image_size")) or None,
        "text_error": safe_str(text_result.get("error")) or None,
        "image_error": safe_str(image_result.get("error")) or None,
        "generated_text": safe_str(text_result.get("generated_text")) or None,
        "image_path": safe_str(image_result.get("image_path")) or None,
        "image_url": safe_str(image_result.get("image_url")) or None,
        "prompt_text": safe_str(text_result.get("prompt")) or None,
        "prompt_image": safe_str(image_result.get("prompt")) or None,
        "base_context_text": (
            safe_str(primary_context_owner.get("base_context"))
            or safe_str(primary_context_owner.get("context"))
            or None
        ),
        "retrieved_context_text": safe_str(primary_retrieved_owner.get("retrieved_context")) or None,
        "retrieved_chunks_json": _json_dump(primary_retrieved_owner.get("retrieved_chunks", [])),
        "entity_snapshot_json": _json_dump(entity),
        "text_result_json": _json_dump(result.get("text_result")),
        "image_result_json": _json_dump(result.get("image_result")),
        "request_options_json": _json_dump(request),
    }

    columns = list(payload.keys())
    placeholders = ", ".join("?" for _ in columns)
    insert_sql = f"INSERT INTO generation_runs ({', '.join(columns)}) VALUES ({placeholders})"

    with _connect(resolved_path) as connection:
        cursor = connection.execute(insert_sql, [payload[column] for column in columns])
        connection.commit()
        return int(cursor.lastrowid)


def count_generation_runs(
    entity_id: str | None = None,
    db_path: Path | None = None,
) -> int:
    """Return the total number of persisted generation runs."""
    resolved_path = db_path or DEFAULT_DB_PATH
    init_store(resolved_path)

    sql = "SELECT COUNT(*) FROM generation_runs"
    params: list[Any] = []
    if entity_id:
        sql += " WHERE entity_id = ?"
        params.append(entity_id)

    with _connect(resolved_path) as connection:
        row = connection.execute(sql, params).fetchone()

    return int(row[0]) if row is not None else 0


def list_generation_runs(
    limit: int | None = 20,
    entity_id: str | None = None,
    db_path: Path | None = None,
) -> list[dict[str, Any]]:
    """List recent generation runs ordered from newest to oldest."""
    resolved_path = db_path or DEFAULT_DB_PATH
    init_store(resolved_path)

    sql = """
    SELECT
        id,
        created_at,
        entity_id,
        entity_name,
        entity_type,
        output_type,
        generate_text,
        generate_image,
        requested_llm_provider,
        effective_text_provider,
        text_mode,
        requested_image_provider,
        effective_image_provider,
        use_llm,
        use_rag,
        embedding_provider,
        image_mode,
        visual_style,
        image_size,
        text_error,
        image_error,
        generated_text,
        image_path,
        image_url
    FROM generation_runs
    """
    params: list[Any] = []
    if entity_id:
        sql += " WHERE entity_id = ?"
        params.append(entity_id)
    sql += " ORDER BY id DESC"
    if limit is not None:
        sql += " LIMIT ?"
        params.append(max(1, int(limit)))

    with _connect(resolved_path) as connection:
        rows = connection.execute(sql, params).fetchall()

    return [_row_to_dict(row) for row in rows]


def get_generation_run(run_id: int, db_path: Path | None = None) -> dict[str, Any] | None:
    """Return a single generation run with parsed JSON payloads."""
    resolved_path = db_path or DEFAULT_DB_PATH
    init_store(resolved_path)

    with _connect(resolved_path) as connection:
        row = connection.execute(
            "SELECT * FROM generation_runs WHERE id = ?",
            (int(run_id),),
        ).fetchone()

    if row is None:
        return None

    return _row_to_dict(row)


def _connect(db_path: Path) -> sqlite3.Connection:
    """Create a SQLite connection configured for dict-like row access."""
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection


def _json_dump(value: Any) -> str | None:
    """Serialize JSON-compatible payloads as UTF-8 text."""
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def _json_load(value: str | None) -> Any:
    """Parse a JSON string when present."""
    if value is None or value == "":
        return None
    return json.loads(value)


def _bool_to_int(value: Any) -> int:
    """Store booleans as SQLite-friendly integers."""
    return 1 if bool(value) else 0


def _maybe_bool(column: str, value: Any) -> Any:
    """Convert SQLite integer flags back to booleans for selected columns."""
    if column in BOOLEAN_COLUMNS:
        return bool(value)
    return value


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    """Convert a SQLite row to a Python dict with parsed JSON and booleans."""
    result: dict[str, Any] = {}

    for key in row.keys():
        value = row[key]
        if key in JSON_COLUMNS:
            result[key] = _json_load(value)
        else:
            result[key] = _maybe_bool(key, value)

    return result
