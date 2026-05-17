"""Tests for the image client without real network calls."""

from __future__ import annotations

from src import image_client


def test_get_available_image_providers_reads_env(monkeypatch) -> None:
    monkeypatch.setattr(image_client, "load_env_file", lambda: None)
    monkeypatch.setenv("OPENAI_API_KEY", "a")

    available = image_client.get_available_image_providers()
    assert available == {"openai": True, "fallback": True}


def test_get_available_image_providers_accepts_runtime_overrides(monkeypatch) -> None:
    monkeypatch.setattr(image_client, "load_env_file", lambda: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    available = image_client.get_available_image_providers(
        api_key_overrides={"openai": "temporary-key"}
    )

    assert available == {"openai": True, "fallback": True}


def test_generate_image_falls_back_when_key_missing(monkeypatch) -> None:
    monkeypatch.setattr(image_client, "load_env_file", lambda: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = image_client.generate_image(provider="openai", prompt="Prompt visual")

    assert result["provider"] == "fallback"
    assert result["status"] == "fallback"
    assert result["prompt"] == "Prompt visual"
    assert result["error"] is not None


def test_generate_image_returns_success_structure(monkeypatch) -> None:
    monkeypatch.setattr(image_client, "load_env_file", lambda: None)
    monkeypatch.setenv("OPENAI_API_KEY", "a")
    monkeypatch.setattr(
        image_client,
        "_generate_with_openai",
        lambda **_: {"image_path": "/tmp/generated.png", "image_url": None},
    )

    result = image_client.generate_image(provider="openai", prompt="Prompt visual")

    assert result == {
        "provider": "openai",
        "status": "success",
        "prompt": "Prompt visual",
        "image_path": "/tmp/generated.png",
        "image_url": None,
        "error": None,
    }


def test_generate_image_uses_runtime_api_key_override(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(image_client, "load_env_file", lambda: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def fake_generate(**kwargs: object) -> dict[str, str | None]:
        captured.update(kwargs)
        return {"image_path": "/tmp/generated.png", "image_url": None}

    monkeypatch.setattr(image_client, "_generate_with_openai", fake_generate)

    result = image_client.generate_image(
        provider="openai",
        prompt="Prompt visual",
        api_key="runtime-openai-key",
    )

    assert result["provider"] == "openai"
    assert captured["api_key"] == "runtime-openai-key"


def test_generate_image_does_not_leak_secrets(monkeypatch) -> None:
    secret = "super-secret-key"
    monkeypatch.setattr(image_client, "load_env_file", lambda: None)
    monkeypatch.setenv("OPENAI_API_KEY", secret)

    def fail(**_: object) -> dict:
        raise RuntimeError(f"boom {secret}")

    monkeypatch.setattr(image_client, "_generate_with_openai", fail)

    result = image_client.generate_image(provider="openai", prompt="Prompt visual")

    assert result["provider"] == "fallback"
    assert secret not in (result["error"] or "")


def test_generate_image_does_not_leak_runtime_override_secret(monkeypatch) -> None:
    secret = "runtime-secret-key"
    monkeypatch.setattr(image_client, "load_env_file", lambda: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def fail(**_: object) -> dict:
        raise RuntimeError(f"boom {secret}")

    monkeypatch.setattr(image_client, "_generate_with_openai", fail)

    result = image_client.generate_image(
        provider="openai",
        prompt="Prompt visual",
        api_key=secret,
    )

    assert result["provider"] == "fallback"
    assert secret not in (result["error"] or "")
