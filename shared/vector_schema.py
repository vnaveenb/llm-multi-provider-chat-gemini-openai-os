"""Dimension-agnostic vector schema utilities.

Embedding dimension is determined at index-creation time by probing the
configured model — not hardcoded — so swapping embedding models only
requires re-indexing, not code changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.embeddings import Embeddings


@dataclass
class VectorIndexConfig:
    """Configuration for a vector index."""

    collection_name: str
    dimension: int | None = None
    distance_metric: str = "cosine"
    metadata_fields: list[str] = field(default_factory=list)

    def resolve_dimension(self, embeddings: Embeddings) -> int:
        """Detect embedding dimension by encoding a probe string. Caches on self."""
        if self.dimension is None:
            probe = embeddings.embed_query("dimension probe")
            self.dimension = len(probe)
        return self.dimension


def build_index_metadata(
    source: str,
    chunk_index: int,
    total_chunks: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build standardised metadata attached to every indexed document."""
    meta: dict[str, Any] = {
        "source": source,
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
    }
    if extra:
        meta.update(extra)
    return meta


__all__ = ["VectorIndexConfig", "build_index_metadata"]
