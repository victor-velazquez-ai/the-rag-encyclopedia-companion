"""ragkit.core — the contracts every other module shares.

Kept deliberately small. Three things live here because the whole library depends on them:

- ``Chunk``        the unit that flows from chunking → embedding → indexing → retrieval (Ch 3).
- ``Config``       the YAML-backed config + the provider registry that makes managed swaps
                   a one-line change (the book's "flip the provider field" promise, Ch 4/7).
- prefixes         the instruction prefixes that MUST match at index and query time, pinned in
                   one place so a mismatch can't silently wreck recall (Ch 4).
"""

# Phase 2 will export: Chunk, Config, load_config, ProviderRegistry, QUERY_PREFIX, DOC_PREFIX
