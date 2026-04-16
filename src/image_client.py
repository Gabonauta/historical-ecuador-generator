"""Provider-agnostic image generation client with safe fallbacks."""

from __future__ import annotations

import base64
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.llm_client import get_safe_error_chain, load_env_file
from src.utils import normalize_text


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMAGE_SAVE_DIR = PROJECT_ROOT / "outputs" / "generated_images"
DEFAULT_IMAGE_MODELS = {
    "openai": "gpt-image-1",
}

IMAGE_ENV_KEYS = {
    "openai": "OPENAI_API_KEY",
}

SUPPORTED_IMAGE_SIZES = (
    "1024x1024",
    "1536x1024",
    "1024x1536",
)

SUPPORTED_IMAGE_QUALITIES = (
    "standard",
    "low",
    "medium",
    "high",
    "auto",
)


class ImageClientError(Exception):
    """Base exception for safe image generation failures."""


class UnsupportedImageProviderError(ImageClientError):
    """Raised when the requested image provider is unsupported."""


class ImageProviderConfigError(ImageClientError):
    """Raised when an image provider has no valid configuration."""


class ImageProviderRequestError(ImageClientError):
    """Raised when the image provider request fails."""


def get_available_image_providers() -> dict[str, bool]:
    """Return which image providers are currently usable."""
    load_env_file()
    return {
        "openai": bool(os.getenv(IMAGE_ENV_KEYS["openai"])),
        "fallback": True,
    }


def generate_image(
    provider: str,
    prompt: str,
    size: str = "1024x1024",
    quality: str = "standard",
    save_dir: str | None = None,
) -> dict[str, Any]:
    """Generate an image using the selected provider or return a safe fallback."""
    load_env_file()
    normalized_provider = normalize_text(provider, lowercase=True) or "fallback"
    normalized_prompt = normalize_text(prompt)
    normalized_size = normalize_text(size) or "1024x1024"
    normalized_quality = normalize_text(quality, lowercase=True) or "standard"

    if not normalized_prompt:
        return _build_fallback_result(
            prompt="",
            error="No se pudo generar imagen porque el prompt visual esta vacio.",
        )

    if normalized_provider == "fallback":
        return _build_fallback_result(prompt=normalized_prompt, error=None)

    if normalized_provider not in IMAGE_ENV_KEYS:
        return _build_fallback_result(
            prompt=normalized_prompt,
            error="Proveedor de imagen no soportado; se devolvio solo el prompt visual.",
        )

    if normalized_size not in SUPPORTED_IMAGE_SIZES:
        return _build_fallback_result(
            prompt=normalized_prompt,
            error="Tamano de imagen no soportado; se devolvio solo el prompt visual.",
        )

    if normalized_quality not in SUPPORTED_IMAGE_QUALITIES:
        return _build_fallback_result(
            prompt=normalized_prompt,
            error="Calidad de imagen no soportada; se devolvio solo el prompt visual.",
        )

    api_key = os.getenv(IMAGE_ENV_KEYS[normalized_provider])
    if not api_key:
        return _build_fallback_result(
            prompt=normalized_prompt,
            error="El proveedor de imagen solicitado no tiene una API key configurada.",
        )

    try:
        if normalized_provider == "openai":
            provider_result = _generate_with_openai(
                api_key=api_key,
                prompt=normalized_prompt,
                size=normalized_size,
                quality=normalized_quality,
                save_dir=save_dir,
            )
        else:
            raise UnsupportedImageProviderError("Proveedor de imagen no soportado.")
    except ImageClientError as error:
        return _build_fallback_result(
            prompt=normalized_prompt,
            error=get_safe_error_chain(error),
        )
    except Exception as error:
        return _build_fallback_result(
            prompt=normalized_prompt,
            error=get_safe_error_chain(error),
        )

    return {
        "provider": normalized_provider,
        "status": "success",
        "prompt": normalized_prompt,
        "image_path": provider_result.get("image_path"),
        "image_url": provider_result.get("image_url"),
        "error": None,
    }


def _generate_with_openai(
    api_key: str,
    prompt: str,
    size: str,
    quality: str,
    save_dir: str | None,
) -> dict[str, Any]:
    """Generate an image using the OpenAI Images API."""
    try:
        from openai import OpenAI
    except ImportError as error:
        raise ImageProviderRequestError("No fue posible cargar el cliente de OpenAI imagen.") from error

    try:
        client = OpenAI(api_key=api_key, timeout=90.0, max_retries=0)
        response = client.images.generate(
            model=DEFAULT_IMAGE_MODELS["openai"],
            prompt=prompt,
            size=size,
            quality=quality,
            output_format="png",
        )
    except Exception as error:
        raise ImageProviderRequestError("Fallo la solicitud al proveedor OpenAI imagen.") from error

    image_path = _extract_and_save_image(response, save_dir)
    image_url = _extract_image_url(response)

    if not image_path and not image_url:
        raise ImageProviderRequestError("El proveedor OpenAI imagen no devolvio una imagen utilizable.")

    return {
        "image_path": image_path,
        "image_url": image_url,
    }


def _extract_and_save_image(response: Any, save_dir: str | None) -> str | None:
    """Save a base64 image when the provider returns inline image bytes."""
    data = getattr(response, "data", None)
    if data is None and isinstance(response, dict):
        data = response.get("data")
    if not data:
        return None

    first_item = data[0]
    image_b64 = getattr(first_item, "b64_json", None)
    if image_b64 is None and isinstance(first_item, dict):
        image_b64 = first_item.get("b64_json")
    if not image_b64:
        return None

    try:
        image_bytes = base64.b64decode(image_b64)
    except Exception as error:
        raise ImageProviderRequestError("La imagen devuelta por el proveedor no pudo decodificarse.") from error

    target_dir = Path(save_dir) if save_dir else DEFAULT_IMAGE_SAVE_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    filename = f"historical_ecuador_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}.png"
    file_path = target_dir / filename
    file_path.write_bytes(image_bytes)
    return str(file_path)


def _extract_image_url(response: Any) -> str | None:
    """Extract an image URL when the provider returns a remote reference."""
    data = getattr(response, "data", None)
    if data is None and isinstance(response, dict):
        data = response.get("data")
    if not data:
        return None

    first_item = data[0]
    image_url = getattr(first_item, "url", None)
    if image_url is None and isinstance(first_item, dict):
        image_url = first_item.get("url")
    return normalize_text(image_url) or None


def _build_fallback_result(prompt: str, error: str | None) -> dict[str, Any]:
    """Return a safe fallback payload preserving the visual prompt."""
    return {
        "provider": "fallback",
        "status": "fallback",
        "prompt": prompt,
        "image_path": None,
        "image_url": None,
        "error": error,
    }
