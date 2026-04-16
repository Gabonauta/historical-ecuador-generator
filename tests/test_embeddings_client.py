"""Tests for the embeddings client without real network calls."""

from __future__ import annotations

import pytest

from src import embeddings_client


def test_get_available_embedding_providers_reads_env(monkeypatch) -> None:
    monkeypatch.setattr(embeddings_client, "load_env_file", lambda: None)
    monkeypatch.setenv("OPENAI_API_KEY", "a")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    available = embeddings_client.get_available_embedding_providers()
    assert available == {"openai": True, "gemini": False}


def test_generate_embedding_raises_controlled_error_when_key_missing(monkeypatch) -> None:
    monkeypatch.setattr(embeddings_client, "load_env_file", lambda: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(embeddings_client.EmbeddingProviderConfigError) as error:
        embeddings_client.generate_embedding("hola", provider="openai")

    assert "API key" in str(error.value)
    assert "OPENAI_API_KEY" not in str(error.value)


def test_generate_embedding_validates_provider() -> None:
    with pytest.raises(embeddings_client.UnsupportedEmbeddingProviderError):
        embeddings_client.generate_embedding("hola", provider="otro")


def test_generate_embeddings_do_not_leak_secrets(monkeypatch) -> None:
    secret = "super-secret-key"
    monkeypatch.setattr(embeddings_client, "load_env_file", lambda: None)
    monkeypatch.setenv("OPENAI_API_KEY", secret)

    def fail(*_: object, **__: object) -> list[list[float]]:
        raise RuntimeError(f"boom {secret}")

    monkeypatch.setattr(embeddings_client, "_generate_with_openai", fail)

    with pytest.raises(embeddings_client.EmbeddingProviderRequestError) as error:
        embeddings_client.generate_embeddings(["hola"], provider="openai")

    assert secret not in str(error.value)
