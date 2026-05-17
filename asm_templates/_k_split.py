"""Shared K-split chunking utility for asm_templates.

Both ``ffn_asm`` and ``projection_asm`` need to split a K-dimension tile count
into chunks of at most ``max_k_tiles``.  This module provides one canonical
implementation so the two callers stay in sync.
"""

from __future__ import annotations


def k_chunks(num_k_tiles: int, max_k_tiles: int) -> list[tuple[int, int]]:
    """Split ``num_k_tiles`` K-dimension tiles into chunks of at most
    ``max_k_tiles``.  Returns a list of ``(k_start_tile, k_count)`` pairs.

    >>> k_chunks(6, 4)
    [(0, 4), (4, 2)]
    >>> k_chunks(4, 4)
    [(0, 4)]
    >>> k_chunks(1, 4)
    [(0, 1)]
    """
    assert max_k_tiles >= 1, f"MAX_K_TILES must be >= 1, got {max_k_tiles}"
    chunks: list[tuple[int, int]] = []
    k_pos = 0
    while k_pos < num_k_tiles:
        count = min(max_k_tiles, num_k_tiles - k_pos)
        chunks.append((k_pos, count))
        k_pos += count
    return chunks
