"""Low-level ISA emission layer for the ATen PLENA compiler."""

from __future__ import annotations

from compiler.asm_templates import (
    layer_norm_asm,
    preload_act_asm,
    preload_addr_reg_asm,
    reset_reg_asm,
    rms_norm_asm,
    rope_asm,
    store_act_asm,
)
from compiler.aten.plena.isa_attention import IsaAttentionMixin
from compiler.aten.plena.isa_emit import IsaEmitMixin
from compiler.aten.plena.isa_fp_ops import IsaFPOpsMixin
from compiler.aten.plena.isa_matrix import IsaMatrixMixin
from compiler.aten.plena.isa_tile_rows import IsaTileRowMixin
from compiler.aten.plena.registers import RegisterAllocator
from compiler.aten.plena.tile_compiler import TileCompiler


class IsaCompiler(
    IsaAttentionMixin,
    IsaMatrixMixin,
    IsaTileRowMixin,
    IsaFPOpsMixin,
    IsaEmitMixin,
    TileCompiler,
):
    """
    ISA Compiler: lowers PLENA compiler operations to assembly text.

    Owns symbol_table, register_allocator, and the InterruptManager.
    Sub-matrix / memory management is inherited from TileCompiler; the
    legacy ``self.tile_compiler`` accessor is preserved as a property
    returning ``self`` for a handful of remaining external callers.
    """

    _ONLINE_SOFTMAX_FPSRAM_BASE = 10

    class InterruptManager:
        """
        Interrupt Manager — manages execution timing only.
        Actual handlers live on IsaCompiler as ``_handle_k_start``,
        ``_handle_k_prefetch_done``, ``_handle_s_tile_done``, ``_handle_k_end``.
        """

        def __init__(self, compiler: IsaCompiler):
            self.compiler = compiler
            self.enabled = False

            self._k_count = 0
            self._tile_count = 0

            self.current_matrix: str = ""
            self.current_activation: str = ""
            self._mlen = compiler.mlen
            self._blen = compiler.blen
            self._batch = compiler.mlen

            self._q_block_idx = 0
            self._k_block_idx = 0
            self._s_tile_address = 0

        @property
        def tile_compiler(self):
            return self.compiler.tile_compiler

        @property
        def k_count(self) -> int:
            return self._k_count

        @property
        def tile_count(self) -> int:
            return self._tile_count

        @property
        def batch(self) -> int:
            return self._batch

        @property
        def out_features(self) -> int:
            if self.current_matrix and self.current_matrix in self:
                info = self[self.current_matrix]
                return info.shape[0]
            return self._mlen

        @property
        def hidden_size(self) -> int:
            if self.current_matrix and self.current_matrix in self:
                info = self[self.current_matrix]
                return info.shape[1]
            return self._mlen

        @property
        def k_block(self) -> int:
            return self._k_block_idx

        @property
        def q_block(self) -> int:
            return self._q_block_idx

        @property
        def s_tile_address(self) -> int:
            return self._s_tile_address

        @property
        def mlen(self) -> int:
            return self._mlen

        @property
        def blen(self) -> int:
            return self._blen

        def reset(self):
            """Reset counters (does not clear current_matrix)."""
            self._k_count = 0
            self._tile_count = 0
            self._q_block_idx = 0
            self._k_block_idx = 0
            self._s_tile_address = 0

        def enable(self):
            self.enabled = True

        def disable(self):
            self.enabled = False

        def trigger_k_start(self) -> str:
            if not self.enabled:
                return ""
            return self.compiler._handle_k_start()

        def trigger_k_prefetch_done(self) -> str:
            if not self.enabled:
                return ""
            result = self.compiler._handle_k_prefetch_done()
            self._k_count += 1
            return result

        def trigger_s_tile_done(self) -> str:
            if not self.enabled:
                return ""
            result = self.compiler._handle_s_tile_done()
            self._tile_count += 1
            return result

        def trigger_k_end(self) -> str:
            if not self.enabled:
                return ""
            return self.compiler._handle_k_end()

    def __init__(self, mlen: int = 64, blen: int = 4, real_data_ratio: float = 1.125, unroll_loops: bool = False):
        # TileCompiler.__init__ sets mlen, blen, unroll_loops, the HBM/VRAM/FPRAM
        # matrices and allocators, loaded_sub_blocks, and _address_cache.
        super().__init__(mlen=mlen, blen=blen, unroll_loops=unroll_loops)
        self.real_data_ratio = real_data_ratio
        self.register_allocator = RegisterAllocator()
        self.generated_code = ""
        self.interrupt = self.InterruptManager(self)

    # Back-compat shim: older callers (and a couple of external ops modules)
    # reach into ``compiler.tile_compiler`` directly. After the merge,
    # IsaCompiler *is* the TileCompiler, so the property just returns
    # ``self``.
    @property
    def tile_compiler(self) -> IsaCompiler:
        return self

    # Interrupt handler placeholders (overridden by flash-attention passes).

    def _handle_k_start(self) -> str:
        return ""

    def _handle_k_prefetch_done(self) -> str:
        return ""

    def _handle_s_tile_done(self) -> str:
        return ""

    def _handle_k_end(self) -> str:
        return ""

    def load_batch(
        self,
        hbm_object_name: str,
        vram_object_name: str,
        vlen: int = 64,
        preload_len: int = 4,
    ) -> str:
        """
        Load a Batch tensor from HBM to VRAM.

        HBM storage is MXFP (1 scale per 8 elements), so HBM actual size =
        logical size * real_data_ratio = 1.125. VRAM stores only the vector
        data (no scale), so VRAM size = logical size.

        Order (matters): allocate VRAM → register in symbol table → emit ISA.
        """
        hbm_layout = self.get_hbm_layout(hbm_object_name)
        h, w = hbm_layout.full_shape
        hbm_addr = hbm_layout.hbm_base_addr
        size = h * w
        vram_base = self.vram_allocator.allocate(size, name=vram_object_name)
        self.add_vram_object(
            name=vram_object_name,
            shape=(h, w),
            vram_addr=vram_base,
            dtype="fp16",
            kind="Batch",
            allocate_if_none=False,
            strict=False,
        )

        addr_reg = self.register_allocator.allocate_addr(1)[0]
        gp_regs_for_addr = self.register_allocator.allocate_gp(1)

        isa_code = f"; Load_Batch {hbm_object_name} -> {vram_object_name}\n"
        isa_code += f"; HBM[{hbm_addr}] → VRAM[{vram_base}], shape=({h}, {w})\n"

        isa_code += preload_addr_reg_asm(
            addr_reg_to_set=[addr_reg], available_registers=gp_regs_for_addr, addr_reg_val=[hbm_addr]
        )

        # preload_act_asm requires 5 GP registers: [a_actual, stride, result, outer_loop, inner_loop].
        gp_regs_for_preload = self.register_allocator.allocate_gp(5)
        isa_code += reset_reg_asm(alive_registers=gp_regs_for_preload)

        isa_code += preload_act_asm(
            vlen=vlen,
            preload_len=preload_len,
            batch=h,
            hidden_size=w,
            alive_registers=gp_regs_for_preload,
            act_vram_offset=vram_base,
            activation_offset_reg=addr_reg,
            stride_size=w,
        )

        self.register_allocator.free_gp(gp_regs_for_addr)
        self.register_allocator.free_gp(gp_regs_for_preload)
        self.register_allocator.free_addr([addr_reg])

        return self._emit(isa_code)

    def store_to_hbm(
        self,
        tensor_name: str,
        hbm_addr: int | None = None,
        hbm_object_name: str | None = None,
        hbm_addr_reg: int | None = None,
        vlen: int = 64,
        precision: int = 0,  # 0 = Activation, 1 = KeyValue
        store_amount: int = 4,  # HBM_V_Writeback_Amount
    ) -> str:
        """
        Write tensor from VRAM back to HBM.

        Used to spill computed intermediates (e.g., K) from VRAM to HBM so
        downstream ops (e.g., QK^T) can read them from HBM. Emits
        ``store_act_asm`` for tensors of any supported size.
        """
        if tensor_name not in self:
            raise KeyError(f"Tensor '{tensor_name}' not found in symbol table")

        tensor_info = self[tensor_name]

        # Batch and VRAMMatrix share the same VRAM storage layout.
        if tensor_info.kind not in ("Batch", "VRAMMatrix"):
            raise ValueError(
                f"Tensor '{tensor_name}' must be Batch or VRAMMatrix to store from VRAM, got {tensor_info.kind}"
            )

        if tensor_info.vram_addr is None:
            raise ValueError(f"Tensor '{tensor_name}' has no VRAM address to store")

        if hbm_addr is None:
            if tensor_info.hbm_addr >= 0:
                hbm_addr = tensor_info.hbm_addr
            else:
                raise ValueError(f"Tensor '{tensor_name}' has no HBM address. Please specify hbm_addr.")

        batch_size = tensor_info.shape[0]
        hidden_size = tensor_info.shape[1]

        isa_code = f"; Store {tensor_name} from VRAM to HBM\n"
        isa_code += f"; VRAM[{tensor_info.vram_addr}] -> HBM[{hbm_addr}], shape=({batch_size}, {hidden_size})\n"

        gp_regs = self.register_allocator.allocate_gp(5)

        if hbm_addr_reg is None:
            addr_regs = self.register_allocator.allocate_addr(1)
            hbm_addr_reg = addr_regs[0]
            need_free_addr = True
        else:
            addr_regs = []
            need_free_addr = False

        try:
            gp_regs_for_addr = self.register_allocator.allocate_gp(2)
            isa_code += preload_addr_reg_asm(
                addr_reg_to_set=[hbm_addr_reg], available_registers=gp_regs_for_addr, addr_reg_val=[hbm_addr]
            )
            self.register_allocator.free_gp(gp_regs_for_addr)

            isa_code += store_act_asm(
                vlen=vlen,
                batch=batch_size,
                hidden_size=hidden_size,
                alive_registers=gp_regs,
                act_vram_offset=tensor_info.vram_addr,
                hbm_addr_reg=hbm_addr_reg,
                stride_size=hidden_size,
                store_amount=store_amount,
            )

            if tensor_info.hbm_addr < 0 or tensor_info.hbm_addr != hbm_addr:
                tensor_info.hbm_addr = hbm_addr
                # HBM stores the MXFP-expanded size (logical size × real_data_ratio).
                size = batch_size * hidden_size
                tensor_info.hbm_size = int(size * self.real_data_ratio)
        finally:
            self.register_allocator.free_gp(gp_regs)
            if need_free_addr:
                self.register_allocator.free_addr(addr_regs)

        if hbm_object_name is not None:
            self.add_hbm_object(
                name=hbm_object_name,
                hbm_addr=hbm_addr,
                shape=(batch_size, hidden_size),
            )

        return self._emit(isa_code)

    def normalize(
        self,
        tensor_name: str,
        mode: str = "rms",
        eps_offset: int = 1,
        reci_hid_offset: int = 2,
        vlen: int | None = None,
        scratchpad_vram_addr: int | None = None,
    ) -> str:
        """
        Normalize a VRAM tensor in-place.

        Supports:
        - mode="rms":   RMSNorm
        - mode="layer": LayerNorm

        Args:
            tensor_name: Tensor name in symbol table (must have VRAM address)
            mode: "rms" or "layer"
            eps_offset: FPRAM address of epsilon
            reci_hid_offset: FPRAM address of 1/hidden_dim
            vlen: vector length (default: self.mlen)
            scratchpad_vram_addr: scratchpad VRAM address (default: auto-allocate temporary space)
        """
        if tensor_name not in self:
            raise KeyError(f"Tensor '{tensor_name}' not found in symbol table")

        tensor_info = self[tensor_name]
        if tensor_info.vram_addr is None:
            raise ValueError(f"Tensor '{tensor_name}' has no VRAM address")

        batch_size, hidden_dim = tensor_info.shape
        if vlen is None:
            vlen = self.mlen
        if hidden_dim % vlen != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by vlen ({vlen}) for normalization_asm")

        mode = mode.lower()
        if mode not in ("rms", "layer"):
            raise ValueError(f"Unsupported normalization mode: {mode}. Expected 'rms' or 'layer'.")

        gp_regs = self.register_allocator.allocate_gp(4)

        temp_scratchpad_name = None
        if scratchpad_vram_addr is None:
            temp_scratchpad_name = f"__norm_scratch__{tensor_name}__{len(self.generated_code)}"
            scratchpad_vram_addr = self.vram_allocator.allocate(vlen, name=temp_scratchpad_name)

        try:
            isa_code = f"; Normalize ({mode}) {tensor_name}, shape=({batch_size}, {hidden_dim})\n"
            if mode == "rms":
                isa_code += rms_norm_asm(
                    _eps_offset=eps_offset,
                    reci_hid_offset=reci_hid_offset,
                    alive_registers=gp_regs,
                    activation_base_address=tensor_info.vram_addr,
                    scratchpad_base_address=scratchpad_vram_addr,
                    vlen=vlen,
                    batch_size=batch_size,
                    hidden_dim=hidden_dim,
                )
            else:
                isa_code += layer_norm_asm(
                    _eps_offset=eps_offset,
                    reci_hid_offset=reci_hid_offset,
                    alive_registers=gp_regs,
                    activation_base_address=tensor_info.vram_addr,
                    scratchpad_base_address=scratchpad_vram_addr,
                    vlen=vlen,
                    batch_size=batch_size,
                    hidden_dim=hidden_dim,
                )

            return self._emit(isa_code)
        finally:
            # Always release allocated GP registers used by normalization template.
            self.register_allocator.free_gp(gp_regs)
            if temp_scratchpad_name is not None:
                self.vram_allocator.free(temp_scratchpad_name, strict=False)

    def rope(
        self,
        x_name: str,
        x_rot_name: str,
        cos_name: str,
        sin_name: str,
    ) -> str:
        """Apply RoPE in-place: x = x * cos + rotate_half(x) * sin

        All four tensors must already be in VRAM with the same shape (seq_len, head_dim).
        x_rot must be preloaded by the caller as rotate_half(x).
        """
        x_info = self[x_name]
        xrot_info = self[x_rot_name]
        cos_info = self[cos_name]
        sin_info = self[sin_name]

        if x_info.vram_addr is None:
            raise ValueError(f"Tensor '{x_name}' has no VRAM address")

        seq_len, head_dim = x_info.shape
        vlen = self.mlen

        if head_dim % vlen != 0:
            raise ValueError(f"head_dim ({head_dim}) must be divisible by vlen ({vlen}) for rope")

        gp_regs = self.register_allocator.allocate_gp(5)

        scratch_name = f"__rope_scratch__{x_name}__{len(self.generated_code)}"
        scratch_addr = self.vram_allocator.allocate(vlen, name=scratch_name)

        try:
            isa_code = rope_asm(
                alive_registers=gp_regs,
                x_base_address=x_info.vram_addr,
                x_rot_base_address=xrot_info.vram_addr,
                cos_base_address=cos_info.vram_addr,
                sin_base_address=sin_info.vram_addr,
                scratchpad_base_address=scratch_addr,
                vlen=vlen,
                seq_len=seq_len,
                head_dim=head_dim,
            )
            return self._emit(isa_code)
        finally:
            self.register_allocator.free_gp(gp_regs)
            self.vram_allocator.free(scratch_name, strict=False)

    def get_code(self) -> str:
        """Get all accumulated generated ISA code"""
        return self.generated_code

    def reset(self):
        """Reset compiler state (clear code, but retain symbol table)"""
        self.generated_code = ""
        self.register_allocator = RegisterAllocator()
        # Call TileCompiler.reset() explicitly since the merged class shadows it.
        TileCompiler.reset(self)

    def print_symbol_table(self):
        """Print symbol table"""
        self.print_table()

    def get_symbol_table(self):
        """Get managed object table view."""
        return self

    def get_tensor_info(self, name: str):
        """Get unified tensor/object info by name."""
        return self[name]

    def add_hbm_object(
        self,
        name: str,
        hbm_addr: int,
        shape: tuple[int, int],
        real_data_ratio: float = 1.125,
    ):
        """Register an HBM object and build its HBM layout.

        Wraps ``TileCompiler.add_hbm_object`` with a different positional
        parameter order ``(name, hbm_addr, shape, ...)`` that all IsaCompiler
        callers use.
        """
        return TileCompiler.add_hbm_object(
            self,
            name=name,
            shape=shape,
            hbm_addr=hbm_addr,
            real_data_ratio=real_data_ratio,
        )

    def free_hbm_object(self, name: str, strict: bool = False):
        """Free an HBM object by name (defaults to non-strict)."""
        return TileCompiler.free_hbm_object(self, name, strict=strict)

    def get_vram_addr(self, name: str) -> int:
        """Get VRAM base address of an object."""
        info = self.get_tensor_info(name)
        if info.vram_addr is None:
            raise ValueError(f"Object '{name}' has no VRAM address")
        return info.vram_addr

    def get_vram_tile_addr(
        self,
        name: str,
        tile_row_idx: int = 0,
        tile_col_idx: int = 0,
    ) -> int:
        """
        Get VRAM address of a specific tile (sub-block) in a VRAM matrix.

        Args:
            name: matrix name
            tile_row_idx: tile row index (0-based)
            tile_col_idx: tile col index (0-based)
        """
        self._ensure_vram_matrix_layout(name)
        sub = self.get_vram_sub_block(name, tile_row_idx, tile_col_idx)
        return sub.vram_addr

    def ensure_hbm_sub_matrix(
        self,
        name: str,
        hbm_addr: int,
        shape: tuple[int, int],
        real_data_ratio: float = 1.125,
    ):
        """Ensure HBM matrix layout exists."""
        if name in self.hbm_matrices:
            return
        self.add_hbm_object(
            name=name,
            hbm_addr=hbm_addr,
            shape=shape,
            real_data_ratio=real_data_ratio,
        )

    def ensure_vram_matrix_layout(self, name: str, shape: tuple[int, int]):
        """Ensure VRAM matrix layout exists for an already allocated VRAM object."""
        if name in self.vram_matrices:
            return
        vram_addr = self.get_vram_addr(name)
        self.add_vram_object(
            name=name,
            shape=shape,
            vram_addr=vram_addr,
            allocate_if_none=False,
            )

    def free_vram_object(self, name: str, strict: bool = False):
        """Free a VRAM object by name (defaults to non-strict)."""
        return TileCompiler.free_vram_object(self, name, strict=strict)


# Compatibility alias for callers that imported the old low-level layer name.
DeveloperCompiler = IsaCompiler


__all__ = ["DeveloperCompiler", "IsaCompiler"]
