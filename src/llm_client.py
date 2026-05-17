"""Multi-provider LLM client with safe secret handling."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOTENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_MODELS = {
    "openai": "gpt-4.1-mini",
    "gemini": "gemini-2.5-flash",
    "xai": "grok-4.20-beta-latest-non-reasoning",
}

ENV_KEYS = {
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "xai": "XAI_API_KEY",
}


class LLMClientError(Exception):
    """Base exception for safe LLM client failures."""


class UnsupportedProviderError(LLMClientError):
    """Raised when an unsupported provider is requested."""


class ProviderConfigError(LLMClientError):
    """Raised when a provider is not configured."""


class ProviderRequestError(LLMClientError):
    """Raised when a provider request fails."""


def get_safe_error_message(
    error: Exception,
    extra_secrets: list[str] | tuple[str, ...] | None = None,
) -> str:
    """Return a sanitized error summary without exposing secrets."""
    message = f"{type(error).__name__}: {error}"
    secrets = [os.getenv(env_key) for env_key in ENV_KEYS.values()]
    if extra_secrets:
        secrets.extend(extra_secrets)
    for secret in secrets:
        if secret:
            message = message.replace(secret, "[REDACTED]")
    return message


def get_safe_error_chain(
    error: Exception,
    extra_secrets: list[str] | tuple[str, ...] | None = None,
) -> str:
    """Return a sanitized summary including the chained root cause when present."""
    parts = [get_safe_error_message(error, extra_secrets=extra_secrets)]
    cause = getattr(error, "__cause__", None)
    while cause:
        parts.append(get_safe_error_message(cause, extra_secrets=extra_secrets))
        cause = getattr(cause, "__cause__", None)
    return " | Causa: ".join(parts)


def load_env_file(path: Path | None = None) -> None:
    """Load environment variables from a local .env file if present."""
    resolved_path = path or DOTENV_PATH
    if not resolved_path.exists():
        return

    for raw_line in resolved_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key and value and key not in os.environ:
            os.environ[key] = value


def get_available_providers(
    api_key_overrides: dict[str, str] | None = None,
) -> dict[str, bool]:
    """Return which configured providers currently have an API key."""
    load_env_file()
    overrides = _normalize_api_key_overrides(api_key_overrides)
    return {
        provider: bool(overrides.get(provider) or os.getenv(env_key))
        for provider, env_key in ENV_KEYS.items()
    }


def generate_text(
    provider: str,
    prompt: str,
    model: str | None = None,
    temperature: float = 0.3,
    api_key: str | None = None,
) -> str:
    """Generate text using the selected provider with safe, provider-specific logic."""
    load_env_file()
    normalized_provider = provider.strip().lower()
    if normalized_provider not in ENV_KEYS:
        raise UnsupportedProviderError("Proveedor LLM no soportado.")

    resolved_api_key = _normalize_api_key(api_key) or os.getenv(ENV_KEYS[normalized_provider])
    if not resolved_api_key:
        raise ProviderConfigError("El proveedor solicitado no tiene una API key configurada.")

    selected_model = model or DEFAULT_MODELS[normalized_provider]

    if normalized_provider == "openai":
        return _generate_with_openai(resolved_api_key, prompt, selected_model, temperature)
    if normalized_provider == "gemini":
        return _generate_with_gemini(resolved_api_key, prompt, selected_model, temperature)
    if normalized_provider == "xai":
        return _generate_with_xai(resolved_api_key, prompt, selected_model, temperature)

    raise UnsupportedProviderError("Proveedor LLM no soportado.")


def _normalize_api_key_overrides(
    api_key_overrides: dict[str, str] | None,
) -> dict[str, str]:
    """Return trimmed provider-specific API key overrides."""
    if not api_key_overrides:
        return {}

    normalized: dict[str, str] = {}
    for provider, raw_value in api_key_overrides.items():
        normalized_provider = provider.strip().lower()
        normalized_key = _normalize_api_key(raw_value)
        if normalized_provider in ENV_KEYS and normalized_key:
            normalized[normalized_provider] = normalized_key
    return normalized


def _normalize_api_key(api_key: str | None) -> str | None:
    """Normalize a runtime API key without persisting it anywhere."""
    if api_key is None:
        return None
    normalized = api_key.strip()
    return normalized or None


def _generate_with_openai(api_key: str, prompt: str, model: str, temperature: float) -> str:
    """Generate text using the OpenAI Python SDK."""
    try:
        from openai import OpenAI
    except ImportError as error:
        raise ProviderRequestError("No fue posible cargar el cliente de OpenAI.") from error

    try:
        client = OpenAI(api_key=api_key, timeout=30.0, max_retries=0)
        response = client.responses.create(
            model=model,
            input=prompt,
            temperature=temperature,
            max_output_tokens=600,
        )
        output_text = getattr(response, "output_text", "") or ""
    except Exception as error:
        raise ProviderRequestError("Fallo la solicitud al proveedor OpenAI.") from error

    if not output_text.strip():
        raise ProviderRequestError("El proveedor OpenAI devolvio una respuesta vacia.")
    return output_text.strip()


def _generate_with_gemini(api_key: str, prompt: str, model: str, temperature: float) -> str:
    """Generate text using the current Gemini Python SDK."""
    try:
        from google import genai
    except ImportError as error:
        raise ProviderRequestError("No fue posible cargar el cliente de Gemini.") from error

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config={"temperature": temperature},
        )
        output_text = getattr(response, "text", "") or ""
    except Exception as error:
        raise ProviderRequestError("Fallo la solicitud al proveedor Gemini.") from error

    if not output_text.strip():
        raise ProviderRequestError("El proveedor Gemini devolvio una respuesta vacia.")
    return output_text.strip()


def _generate_with_xai(api_key: str, prompt: str, model: str, temperature: float) -> str:
    """Generate text using xAI's OpenAI-compatible API."""
    try:
        from openai import OpenAI
    except ImportError as error:
        raise ProviderRequestError("No fue posible cargar el cliente de xAI.") from error

    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1",
            timeout=30.0,
            max_retries=0,
        )
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": "Responde solo con el texto solicitado."},
                {"role": "user", "content": prompt},
            ],
        )
        output_text = response.choices[0].message.content or ""
    except Exception as error:
        raise ProviderRequestError("Fallo la solicitud al proveedor xAI.") from error

    if not output_text.strip():
        raise ProviderRequestError("El proveedor xAI devolvio una respuesta vacia.")
    return output_text.strip()
