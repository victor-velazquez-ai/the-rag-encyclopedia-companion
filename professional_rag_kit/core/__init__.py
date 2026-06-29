"""professional_rag_kit.core — the contracts every other module shares.

- ``Chunk``            the unit that flows chunking → embedding → indexing → retrieval (Ch 3).
- ``Config`` / ``load_config`` / ``ProviderRegistry``  the bring-your-own-key provider swap (Ch 4/7).
"""

from professional_rag_kit.core.config import Config, ProviderRegistry, load_config
from professional_rag_kit.core.schema import Chunk

__all__ = ["Chunk", "Config", "load_config", "ProviderRegistry"]
