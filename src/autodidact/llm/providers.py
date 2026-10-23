"""Supported providers and how to build a client from settings.

Both providers are reached through their OpenAI-compatible endpoints, so the
same client code serves both. Defaults target the free tiers; every value can
be overridden in ``.env`` (see ``.env.example``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from autodidact.config import env
from autodidact.llm.cache import ResponseCache
from autodidact.llm.openai_compat import OpenAICompatLLM


@dataclass(frozen=True)
class ProviderSpec:
    base_url: str
    key_env: str
    default_model: str
    default_rpm: float
    default_reasoning_effort: str | None
    signup_url: str


PROVIDERS = {
    "gemini": ProviderSpec(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        key_env="GEMINI_API_KEY",
        default_model="gemini-3.5-flash-lite",
        default_rpm=10,
        default_reasoning_effort="low",
        signup_url="https://aistudio.google.com",
    ),
    "groq": ProviderSpec(
        base_url="https://api.groq.com/openai/v1",
        key_env="GROQ_API_KEY",
        default_model="openai/gpt-oss-120b",
        default_rpm=20,
        default_reasoning_effort="low",
        signup_url="https://console.groq.com",
    ),
}

DEFAULT_PROVIDER = "gemini"


class ConfigError(Exception):
    pass


def resolve(provider: str | None = None, model: str | None = None) -> tuple[str, ProviderSpec, str]:
    provider = (provider or env("AUTODIDACT_PROVIDER", DEFAULT_PROVIDER)).lower()
    if provider not in PROVIDERS:
        raise ConfigError(f"unknown provider '{provider}' (choose: {', '.join(PROVIDERS)})")
    spec = PROVIDERS[provider]
    model = model or env(f"{provider.upper()}_MODEL", spec.default_model)
    return provider, spec, model


def make_llm(
    provider: str | None = None,
    model: str | None = None,
    *,
    cache_dir: str | Path | None = "cache",
    **overrides,
) -> OpenAICompatLLM:
    """Build a client from ``.env`` settings. Raises ``ConfigError`` if the key is missing."""
    provider, spec, model = resolve(provider, model)
    key = env(spec.key_env)
    if not key:
        raise ConfigError(
            f"{spec.key_env} is not set. Create a key at {spec.signup_url} "
            "and put it in your .env file."
        )
    prefix = provider.upper()
    temperature = env(f"{prefix}_TEMPERATURE")
    options = {
        "rpm": float(env(f"{prefix}_RPM", str(spec.default_rpm))),
        "temperature": float(temperature) if temperature else None,
        "reasoning_effort": env(f"{prefix}_REASONING_EFFORT", spec.default_reasoning_effort),
        "cache": ResponseCache(Path(cache_dir) / "llm.sqlite3") if cache_dir else None,
    }
    if options["reasoning_effort"] == "none":
        options["reasoning_effort"] = None
    options.update(overrides)
    return OpenAICompatLLM(
        provider=provider, base_url=spec.base_url, api_key=key, model=model, **options
    )
