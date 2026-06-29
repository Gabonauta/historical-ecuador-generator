"""Tests for local SQLite persistence of generation history."""

from __future__ import annotations

from pathlib import Path

from src.history_store import (
    count_generation_runs,
    get_generation_run,
    init_store,
    list_generation_runs,
    save_generation_run,
)


def build_entity() -> dict:
    return {
        "id": "eugenio_espejo",
        "nombre": "Eugenio Espejo",
        "tipo": "personaje",
        "epoca": "Siglo XVIII",
        "ubicacion": "Quito, Ecuador",
    }


def build_request(*, generate_text: bool = True, generate_image: bool = False, use_rag: bool = False) -> dict:
    return {
        "output_type": "ficha_historica",
        "llm_provider": "openai",
        "image_provider": "openai",
        "embedding_provider": "openai",
        "use_llm": True,
        "use_rag": use_rag,
        "top_k": 5,
        "generate_text": generate_text,
        "generate_image": generate_image,
        "image_mode": "retrato_historico",
        "visual_style": "realista",
        "image_size": "1024x1024",
        "debug_mode": False,
    }


def build_text_result(*, mode: str = "fallback", error: str | None = None, use_rag: bool = False) -> dict:
    return {
        "mode": mode,
        "provider": "fallback" if mode == "fallback" else "gemini",
        "output_type": "ficha_historica",
        "use_rag": use_rag,
        "embedding_provider": "openai",
        "prompt": "PROMPT TEXTUAL",
        "base_context": "CONTEXTO BASE",
        "context": "CONTEXTO BASE",
        "retrieved_context": "CONTEXTO RECUPERADO" if use_rag else "",
        "retrieved_chunks": [{"nombre": "Chunk 1", "texto": "Dato"}] if use_rag else [],
        "generated_text": "Texto generado",
        "error": error,
    }


def build_image_result(*, error: str | None = None) -> dict:
    return {
        "provider": "openai",
        "status": "success",
        "image_mode": "retrato_historico",
        "visual_style": "realista",
        "size": "1024x1024",
        "use_rag": True,
        "embedding_provider": "openai",
        "prompt": "PROMPT VISUAL",
        "base_context": "CONTEXTO BASE VISUAL",
        "retrieved_context": "CONTEXTO RECUPERADO VISUAL",
        "retrieved_chunks": [{"nombre": "Chunk visual", "texto": "Dato visual"}],
        "image_path": "outputs/generated_images/test.png",
        "image_url": None,
        "error": error,
    }


def test_init_store_creates_sqlite_file(tmp_path: Path) -> None:
    db_path = tmp_path / "history.sqlite3"
    init_store(db_path)
    assert db_path.exists()


def test_save_and_get_text_only_run(tmp_path: Path) -> None:
    db_path = tmp_path / "history.sqlite3"
    result = {
        "text_result": build_text_result(mode="fallback", error="Fallo seguro", use_rag=True),
        "image_result": None,
        "entity_id": "eugenio_espejo",
        "output_type": "ficha_historica",
        "generate_text": True,
        "generate_image": False,
    }

    run_id = save_generation_run(
        build_entity(),
        build_request(generate_text=True, generate_image=False, use_rag=True),
        result,
        db_path=db_path,
    )
    stored = get_generation_run(run_id, db_path=db_path)

    assert stored is not None
    assert stored["entity_id"] == "eugenio_espejo"
    assert stored["text_mode"] == "fallback"
    assert stored["generate_image"] is False
    assert stored["image_path"] is None
    assert stored["image_url"] is None
    assert stored["use_rag"] is True
    assert stored["retrieved_context_text"] == "CONTEXTO RECUPERADO"
    assert stored["retrieved_chunks_json"] == [{"nombre": "Chunk 1", "texto": "Dato"}]
    assert stored["text_result_json"]["generated_text"] == "Texto generado"


def test_save_and_get_multimodal_run(tmp_path: Path) -> None:
    db_path = tmp_path / "history.sqlite3"
    result = {
        "text_result": build_text_result(mode="llm", error=None, use_rag=False),
        "image_result": build_image_result(),
        "entity_id": "eugenio_espejo",
        "output_type": "ficha_historica",
        "generate_text": True,
        "generate_image": True,
    }

    run_id = save_generation_run(
        build_entity(),
        build_request(generate_text=True, generate_image=True, use_rag=False),
        result,
        db_path=db_path,
    )
    stored = get_generation_run(run_id, db_path=db_path)

    assert stored is not None
    assert stored["effective_text_provider"] == "gemini"
    assert stored["effective_image_provider"] == "openai"
    assert stored["image_path"] == "outputs/generated_images/test.png"
    assert stored["prompt_image"] == "PROMPT VISUAL"
    assert stored["image_result_json"]["status"] == "success"


def test_list_generation_runs_returns_newest_first(tmp_path: Path) -> None:
    db_path = tmp_path / "history.sqlite3"
    entity = build_entity()

    first_result = {
        "text_result": build_text_result(),
        "image_result": None,
        "entity_id": "eugenio_espejo",
        "output_type": "ficha_historica",
        "generate_text": True,
        "generate_image": False,
    }
    second_result = {
        "text_result": build_text_result(mode="llm"),
        "image_result": None,
        "entity_id": "eugenio_espejo",
        "output_type": "ficha_historica",
        "generate_text": True,
        "generate_image": False,
    }

    first_id = save_generation_run(entity, build_request(), first_result, db_path=db_path)
    second_id = save_generation_run(entity, build_request(), second_result, db_path=db_path)

    runs = list_generation_runs(limit=10, db_path=db_path)

    assert [run["id"] for run in runs] == [second_id, first_id]


def test_list_generation_runs_supports_entity_filter(tmp_path: Path) -> None:
    db_path = tmp_path / "history.sqlite3"

    entity_a = build_entity()
    entity_b = {**build_entity(), "id": "plaza_grande", "nombre": "Plaza Grande", "tipo": "lugar"}

    result = {
        "text_result": build_text_result(),
        "image_result": None,
        "entity_id": "eugenio_espejo",
        "output_type": "ficha_historica",
        "generate_text": True,
        "generate_image": False,
    }

    save_generation_run(entity_a, build_request(), result, db_path=db_path)
    save_generation_run(
        entity_b,
        build_request(),
        {**result, "entity_id": "plaza_grande"},
        db_path=db_path,
    )

    runs = list_generation_runs(limit=10, entity_id="plaza_grande", db_path=db_path)

    assert len(runs) == 1
    assert runs[0]["entity_id"] == "plaza_grande"


def test_count_generation_runs_returns_total_and_filtered_total(tmp_path: Path) -> None:
    db_path = tmp_path / "history.sqlite3"

    entity_a = build_entity()
    entity_b = {**build_entity(), "id": "plaza_grande", "nombre": "Plaza Grande", "tipo": "lugar"}
    result = {
        "text_result": build_text_result(),
        "image_result": None,
        "entity_id": "eugenio_espejo",
        "output_type": "ficha_historica",
        "generate_text": True,
        "generate_image": False,
    }

    save_generation_run(entity_a, build_request(), result, db_path=db_path)
    save_generation_run(
        entity_b,
        build_request(),
        {**result, "entity_id": "plaza_grande"},
        db_path=db_path,
    )

    assert count_generation_runs(db_path=db_path) == 2
    assert count_generation_runs(entity_id="plaza_grande", db_path=db_path) == 1


def test_list_generation_runs_without_limit_returns_all_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "history.sqlite3"
    entity = build_entity()
    result = {
        "text_result": build_text_result(),
        "image_result": None,
        "entity_id": "eugenio_espejo",
        "output_type": "ficha_historica",
        "generate_text": True,
        "generate_image": False,
    }

    first_id = save_generation_run(entity, build_request(), result, db_path=db_path)
    second_id = save_generation_run(
        entity,
        build_request(),
        {**result, "text_result": build_text_result(mode="llm")},
        db_path=db_path,
    )

    runs = list_generation_runs(limit=None, db_path=db_path)

    assert [run["id"] for run in runs] == [second_id, first_id]
