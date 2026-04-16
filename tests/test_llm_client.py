"""Tests for the multi-provider LLM client."""

from pathlib import Path

import pytest

from src import llm_client


def test_get_available_providers_reads_env(monkeypatch) -> None:
    monkeypatch.setattr(llm_client, "DOTENV_PATH", Path("/tmp/nonexistent-phase3.env"))
    monkeypatch.setenv("OPENAI_API_KEY", "a")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("XAI_API_KEY", "b")

    available = llm_client.get_available_providers()
    assert available == {"openai": True, "gemini": False, "xai": True}


def test_generate_text_raises_controlled_error_when_key_missing(monkeypatch) -> None:
    monkeypatch.setattr(llm_client, "DOTENV_PATH", Path("/tmp/nonexistent-phase3.env"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(llm_client.ProviderConfigError) as error:
        llm_client.generate_text(provider="openai", prompt="hola")

    assert "API key" in str(error.value)
    assert "OPENAI_API_KEY" not in str(error.value)


def test_generate_text_validates_provider(monkeypatch) -> None:
    with pytest.raises(llm_client.UnsupportedProviderError):
        llm_client.generate_text(provider="otro", prompt="hola")


def test_error_messages_do_not_expose_secrets(monkeypatch) -> None:
    secret = "super-secret-key"
    monkeypatch.setenv("XAI_API_KEY", secret)

    def fake_generate(*_: object, **__: object) -> str:
        raise llm_client.ProviderRequestError("Fallo la solicitud al proveedor xAI.")

    monkeypatch.setattr(llm_client, "_generate_with_xai", fake_generate)

    with pytest.raises(llm_client.ProviderRequestError) as error:
        llm_client.generate_text(provider="xai", prompt="hola")

    assert secret not in str(error.value)
