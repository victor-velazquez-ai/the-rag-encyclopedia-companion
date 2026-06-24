"""Tests for core contracts: Chunk schema, Config/provider swap, and grounded-answer parsing."""

import pytest

from ragkit.core import Chunk, Config, ProviderRegistry, load_config
from ragkit.production.generation import ABSTENTION, GroundedGenerator


def test_chunk_defaults_and_contextual_prefix():
    c = Chunk(id="c1", text="The tax rate is 21%.", doc_id="d1")
    assert c.allowed_groups == [] and c.section_path == []
    ctx = c.with_context("From the 2024 10-K, Tax section:")
    assert ctx.text.startswith("From the 2024 10-K")
    assert ctx.tags["original_text"] == "The tax rate is 21%."  # raw preserved for citation


def test_default_config_is_byo_api_stack():
    cfg = Config()
    assert cfg.generation.provider == "anthropic"
    assert cfg.generation.model == "claude-opus-4-8"
    assert cfg.embedding.provider == "openai"  # Anthropic has no embeddings API
    assert cfg.rerank.provider == "llm"


def test_env_overrides_provider(monkeypatch):
    monkeypatch.setenv("GEN_PROVIDER", "openai")
    monkeypatch.setenv("EMBED_PROVIDER", "voyage")
    cfg = load_config()
    assert cfg.generation.provider == "openai"
    assert cfg.embedding.provider == "voyage"


def test_key_for_reads_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert Config().key_for("generation") == "sk-test"


def test_provider_registry_has_generation_backends():
    # registered by importing ragkit.production.generation
    assert ProviderRegistry.get("generation", "anthropic")
    assert ProviderRegistry.get("generation", "openai")


def test_provider_registry_unknown_raises():
    with pytest.raises(ValueError):
        ProviderRegistry.get("generation", "nope")


def test_grounded_answer_parses_citations_and_abstention():
    # exercise the pure parsing path with a fake backend (no API key needed)
    ProviderRegistry.register("generation", "_fake_cited")(
        lambda model, system, prompt, max_tokens: "The rate is 21% [2] and rising [3]."
    )
    gen = GroundedGenerator(provider="_fake_cited", model="x")
    ans = gen.generate("rate?", ["p0", "p1", "p2"])
    assert ans.citations == [2, 3]
    assert ans.abstained is False

    ProviderRegistry.register("generation", "_fake_abstain")(
        lambda model, system, prompt, max_tokens: ABSTENTION
    )
    ans2 = GroundedGenerator(provider="_fake_abstain", model="x").generate("rate?", ["p0"])
    assert ans2.abstained is True
