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


def get_safe_error_message(error: Exception) -> str:
    """Return a sanitized error summary without exposing secrets."""
    message = f"{type(error).__name__}: {error}"
    for env_key in ENV_KEYS.values():
        secret = os.getenv(env_key)
        if secret:
            message = message.replace(secret, "[REDACTED]")
    return message


def get_safe_error_chain(error: Exception) -> str:
    """Return a sanitized summary including the chained root cause when present."""
    parts = [get_safe_error_message(error)]
    cause = getattr(error, "__cause__", None)
    while cause:
        parts.append(get_safe_error_message(cause))
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


def get_available_providers() -> dict[str, bool]:
    """Return which configured providers currently have an API key."""
    load_env_file()
    return {provider: bool(os.getenv(env_key)) for provider, env_key in ENV_KEYS.items()}


def generate_text(
    provider: str,
    prompt: str,
    model: str | None = None,
    temperature: float = 0.3,
) -> str:
    """Generate text using the selected provider with safe, provider-specific logic."""
    load_env_file()
    normalized_provider = provider.strip().lower()
    if normalized_provider not in ENV_KEYS:
        raise UnsupportedProviderError("Proveedor LLM no soportado.")

    api_key = os.getenv(ENV_KEYS[normalized_provider])
    if not api_key:
        raise ProviderConfigError("El proveedor solicitado no tiene una API key configurada.")

    selected_model = model or DEFAULT_MODELS[normalized_provider]

    if normalized_provider == "openai":
        return _generate_with_openai(api_key, prompt, selected_model, temperature)
    if normalized_provider == "gemini":
        return _generate_with_gemini(api_key, prompt, selected_model, temperature)
    if normalized_provider == "xai":
        return _generate_with_xai(api_key, prompt, selected_model, temperature)

    raise UnsupportedProviderError("Proveedor LLM no soportado.")


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
