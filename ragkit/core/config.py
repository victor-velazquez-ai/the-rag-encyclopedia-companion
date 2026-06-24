"""Config + the provider registry — what makes a managed swap a one-line change (Book Ch 4/7).

Every swappable component (embedder, reranker, generator) is built through a factory that reads a
``provider`` name from YAML or the environment. The default config selects the all-open stack;
``configs/managed.yaml`` selects managed providers. The pipeline code never changes — only the
provider does. See docs/PROVIDER-SWAP.md.

Phase-1 scaffold: the intended surface is sketched below; implemented in Phase 2. Stays importable.
"""

# --- Phase-2 target (spec) ----------------------------------------------------
# def load_config(path: str | None = None) -> "Config":
#     """Load configs/default.yaml (all-open) unless RAGKIT_CONFIG / path overrides."""
#
# class ProviderRegistry:
#     """Maps ('embedding'|'rerank'|'generation', provider_name) -> a constructor.
#     Components call this instead of importing a vendor SDK directly, so 'flip the
#     provider field' is the whole change."""

__all__ = ["Config", "load_config", "ProviderRegistry"]  # populated in Phase 2
