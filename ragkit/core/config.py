"""Config + the provider registry — what makes a managed swap a one-line change (Book Ch 4/7).

The companion runs on **bring-your-own-key cloud APIs** so every example works on a laptop with
no GPU. Each swappable component (generation, embedding, rerank) is built through a factory that
reads a ``provider`` name from YAML or the environment. The default config selects the API stack;
self-hosting the book's verdict picks (Qwen3 / jina) is a documented swap.

    generation : "anthropic" (claude-opus-4-8, default) | "openai" (gpt)
    embedding  : "openai" (text-embedding-3-large, default) | "voyage"   [Anthropic has no embeddings API]
    rerank     : "llm" (listwise, reuses the generation provider, default) | "cohere"

The pipeline code never changes — only the provider does. See docs/PROVIDER-SWAP.md.
"""

from __future__ import annotations

import os
from typing import Any, Callable

from pydantic import BaseModel, Field

# --- Defaults (bring-your-own-key API stack) ---------------------------------
DEFAULTS: dict[str, dict[str, str]] = {
    "generation": {"provider": "anthropic", "model": "claude-opus-4-8"},
    "embedding": {"provider": "openai", "model": "text-embedding-3-large"},
    "rerank": {"provider": "llm", "model": ""},  # "" → reuse the generation model
}

# Env var that names the API key for each provider (the only secret you supply).
PROVIDER_KEYS: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "voyage": "VOYAGE_API_KEY",
    "cohere": "COHERE_API_KEY",
}


class ComponentConfig(BaseModel):
    provider: str
    model: str = ""


class Config(BaseModel):
    """The resolved pipeline config. ``load_config`` builds it from YAML + env overrides."""

    generation: ComponentConfig = Field(
        default_factory=lambda: ComponentConfig(**DEFAULTS["generation"])
    )
    embedding: ComponentConfig = Field(
        default_factory=lambda: ComponentConfig(**DEFAULTS["embedding"])
    )
    rerank: ComponentConfig = Field(default_factory=lambda: ComponentConfig(**DEFAULTS["rerank"]))

    def key_for(self, component: str) -> str | None:
        """Return the API key for a component's provider, read from the environment."""
        provider = getattr(self, component).provider
        env = PROVIDER_KEYS.get(provider)
        return os.environ.get(env) if env else None


def load_config(path: str | None = None) -> Config:
    """Load a config from YAML (if given / RAGKIT_CONFIG set), then apply env overrides.

    Env overrides (so you never have to edit a file to swap a provider):
        EMBED_PROVIDER, GEN_PROVIDER, RERANK_PROVIDER  — the provider name
    """
    data: dict[str, Any] = {}
    path = path or os.environ.get("RAGKIT_CONFIG")
    if path and os.path.exists(path):
        import yaml  # local import: pure-config callers needn't have pyyaml

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

    cfg = Config(**data)

    # env overrides win over the file
    for component, env in (
        ("generation", "GEN_PROVIDER"),
        ("embedding", "EMBED_PROVIDER"),
        ("rerank", "RERANK_PROVIDER"),
    ):
        val = os.environ.get(env)
        if val:
            getattr(cfg, component).provider = val
    return cfg


class ProviderRegistry:
    """Maps (component, provider_name) -> a constructor.

    Components call ``ProviderRegistry.get(...)`` instead of importing a vendor SDK directly, so
    "flip the provider field" is the whole change. Register a backend with the ``@register``
    decorator next to its implementation.
    """

    _registry: dict[tuple[str, str], Callable[..., Any]] = {}

    @classmethod
    def register(cls, component: str, provider: str) -> Callable[[Callable], Callable]:
        def deco(fn: Callable) -> Callable:
            cls._registry[(component, provider)] = fn
            return fn

        return deco

    @classmethod
    def get(cls, component: str, provider: str) -> Callable[..., Any]:
        try:
            return cls._registry[(component, provider)]
        except KeyError:
            known = sorted(p for (c, p) in cls._registry if c == component)
            raise ValueError(
                f"No {component} provider '{provider}'. Registered: {known or '(none yet)'}. "
                f"See docs/PROVIDER-SWAP.md."
            ) from None


__all__ = ["Config", "ComponentConfig", "load_config", "ProviderRegistry", "DEFAULTS", "PROVIDER_KEYS"]
