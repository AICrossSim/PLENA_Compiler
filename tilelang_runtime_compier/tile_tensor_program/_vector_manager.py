"""VectorManager: vector objects, vector tiles, and FP-fragment backing."""

from __future__ import annotations

from math import ceil
from typing import Dict, List, Optional

from ._types import *  # noqa: F401,F403
from ._helpers import *  # noqa: F401,F403


class VectorManager:
    """Own vector-specific logical/runtime behavior.

    Today vectors are FP-fragment-backed rather than ValueTile-backed. This
    manager centralizes:

    - logical `Vector` creation and registration
    - `VectorTile` creation
    - `VectorTile -> FPFragment` binding and lookup
    - eager vector backing initialization
    - vector FP-var resolution helpers used by `mapf`
    """

    def __init__(self, program: "TileTensorProgram") -> None:
        self.program = program
        self.fp_fragment_bindings: Dict[str, str] = {}

    def vector(self, name: str, logical_shape: LogicalShape) -> Vector:
        if len(logical_shape) != 3:
            raise ValueError(f"vector expects one 3D logical shape, got {logical_shape}")
        vector = Vector(program=self.program, name=name, logical_shape=logical_shape)
        self.program.tensor_manager.vectors[name] = vector
        return vector

    def create_vector_tiles(self, vector_name: str, logical_shape: LogicalShape) -> Dict[TileCoord, VectorTile]:
        rows, cols = _logical_shape_to_physical_shape(logical_shape)
        row_blocks = ceil(rows / self.program.mlen)
        col_blocks = ceil(cols / self.program.mlen)
        tiles: Dict[TileCoord, VectorTile] = {}
        tensor_manager = self.program.tensor_manager
        for row_block in range(row_blocks):
            for col_block in range(col_blocks):
                row_count = min(self.program.mlen, rows - row_block * self.program.mlen)
                col_count = min(self.program.mlen, cols - col_block * self.program.mlen)
                vector_tile = VectorTile(
                    tile_id=tensor_manager._next_tensor_tile_id(),
                    tensor_name=vector_name,
                    coord=(row_block, col_block),
                    tile_shape=(row_count, col_count),
                    metadata=tensor_manager._build_tile_metadata(
                        logical_shape,
                        row_block,
                        col_block,
                        row_count,
                        col_count,
                    ),
                )
                tiles[(row_block, col_block)] = vector_tile
                tensor_manager.vector_tiles[vector_tile.tile_id] = vector_tile
                tensor_manager.tensor_tiles[vector_tile.tile_id] = vector_tile
        return tiles

    def bind_tile_to_fp_fragment(self, tile: VectorTile, fragment: FPFragment) -> FPFragment:
        self.fp_fragment_bindings[tile.tile_id] = fragment.name
        return fragment

    def resolve_fp_fragment(self, tile: VectorTile) -> FPFragment:
        fragment_name = self.fp_fragment_bindings.get(tile.tile_id)
        if not isinstance(fragment_name, str):
            raise RuntimeError(f"VectorTile {tile.tile_id} is not bound to one FPFragment")
        fragment = self.program.tensor_manager.fp_fragments.get(fragment_name)
        if not isinstance(fragment, FPFragment):
            raise RuntimeError(
                f"VectorTile {tile.tile_id} binding points to missing FPFragment {fragment_name!r}"
            )
        return fragment

    def initialize_vector_backing(self, vector: Vector, *, init_zero: bool = False) -> None:
        for tile in _tiles_in_grid_order(vector.tiles):
            if self.fp_fragment_bindings.get(tile.tile_id):
                continue
            fragment_name = self.program._auto_name(f"{vector.name}.fp_tile")
            fragment = self.program.tensor_manager.fp_fragment(
                name=fragment_name,
                shape=tile.tile_shape,
                init=0.0,
            )
            self.bind_tile_to_fp_fragment(tile, fragment)
            tile.metadata["fp_fragment_name"] = fragment.name
            if init_zero:
                self.program.fp_fill(fragment, 0.0)

    def resolve_vector_fp_vars(self, vector: Vector) -> List[FPVar]:
        resolved: List[FPVar] = []
        for logical_index in _iter_logical_indices(vector.logical_shape):
            resolved.append(self.program.tensor_manager._resolve_element_fpvar(ElementRef(base=vector, indices=logical_index)))
        return resolved

    def resolve_vector_slice_fp_vars(self, vector_slice: VectorSlice) -> List[FPVar]:
        resolved: List[FPVar] = []
        for logical_index in _iter_selected_logical_indices(vector_slice.base.logical_shape, vector_slice.selectors):
            resolved.append(
                self.program.tensor_manager._resolve_element_fpvar(
                    ElementRef(base=vector_slice.base, indices=logical_index)
                )
            )
        return resolved

    def resolve_vector_tile_fp_vars(self, tile: VectorTile) -> List[FPVar]:
        fragment = self.resolve_fp_fragment(tile)
        row_groups = _vector_tile_row_fp_groups(
            src_tile=tile,
            fragment=fragment,
            mlen=self.program.mlen,
            btmm_hlen=self.program.btmm_hlen,
            src_slice_ranges=None,
        )
        return [fp_var for row in row_groups for fp_var in row]

