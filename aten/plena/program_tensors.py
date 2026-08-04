"""Tensor, memory, and normalization operations for the PLENA program builder."""

from __future__ import annotations

from compiler.asm_templates import ffn_asm, preload_addr_reg_asm, reset_reg_asm
from compiler.aten.plena.packed_kv import (
    PackedKVAppendAddress,
    PackedKVLayout,
    resolve_packed_kv_append,
)
from compiler.aten.plena.vars import FPVar, InputVar, TensorVar, VRAMMatrixVar


class ProgramTensorMixin:
    # ========================================================================
    # Input Declaration
    # ========================================================================

    def input(
        self,
        name: str,
        shape: tuple[int, int],
        hbm_addr: int | None = None,
        prestaged_vram_addr: int | None = None,
        physical_shape: tuple[int, int] | None = None,
        hbm_row_width: int | None = None,
        hbm_element_width: int | None = None,
        hbm_block_size: int | None = None,
        hbm_scale_width: int | None = None,
        precision_role: str = "activation",
    ) -> InputVar:
        """
        Declare an input tensor (in HBM).

        Args:
            name: tensor name
            shape: (height, width)
            hbm_addr: HBM address (None = auto-allocate)
            prestaged_vram_addr: If an int, the tensor is assumed to be already
                present in VRAM at this byte address.  A subsequent call to
                ``load_batch`` will register it at that address without emitting
                any HBM→VRAM prefetch instructions.  If None (default), the
                normal HBM→VRAM load path is used.

        Returns:
            InputVar proxy object
        """
        h, w = physical_shape or shape
        size = h * w
        hbm_size = self.hbm_tensor_size(
            size,
            hbm_row_width=hbm_row_width,
            hbm_element_width=hbm_element_width,
            hbm_block_size=hbm_block_size,
            hbm_scale_width=hbm_scale_width,
        )

        if hbm_addr is None:
            hbm_addr = self._allocate_hbm(hbm_size)

        var = InputVar(
            self,
            name,
            shape,
            hbm_addr,
            hbm_size,
            prestaged_vram_addr=prestaged_vram_addr,
            physical_shape=physical_shape,
        )
        self._inputs[name] = var
        super().add_hbm_object(
            name=name,
            hbm_addr=hbm_addr,
            shape=shape,
            physical_shape=physical_shape,
            real_data_ratio=self.real_data_ratio,
            hbm_row_width=hbm_row_width,
            hbm_element_width=hbm_element_width,
            hbm_block_size=hbm_block_size,
            hbm_scale_width=hbm_scale_width,
            precision_role=precision_role,
        )
        return var

    # ========================================================================
    # Load Operations
    # ========================================================================

    def load_batch(
        self,
        input_var: InputVar,
        name: str | None = None,
    ) -> VRAMMatrixVar:
        """
        Load tensor from HBM to VRAM (Batch type).

        When ``input_var.prestaged_vram_addr`` is set the tensor is assumed to
        be already resident in VRAM at that address.  No HBM→VRAM prefetch
        instructions are emitted; the tensor is simply registered in the symbol
        table at the given address.

        Args:
            input_var: source InputVar
            name: result name (None = use input name)

        Returns:
            VRAMMatrixVar proxy object
        """
        if not isinstance(input_var, InputVar):
            raise TypeError(f"Expected InputVar, got {type(input_var)}")

        display_name = name if name is not None else input_var.display_name
        internal_name = self._scoped_name(display_name)

        if input_var.prestaged_vram_addr is not None:
            # Prestaged path: tensor is already in VRAM — register without ISA.
            h, w = input_var.physical_shape
            vram_addr = input_var.prestaged_vram_addr
            # Tell the VRAM allocator that this region is occupied so subsequent
            # allocations don't collide with it.
            self.vram_allocator._vmm.mark_used(vram_addr, h * w, name=internal_name)
            super().add_vram_object(
                name=internal_name,
                shape=input_var.shape,
                physical_shape=input_var.physical_shape,
                vram_addr=vram_addr,
                dtype="fp16",
                kind="Batch",
                allocate_if_none=False,
                strict=False,
            )
        else:
            # Normal path: emit HBM → VRAM prefetch ISA.
            super().load_batch(
                hbm_object_name=input_var.name,
                vram_object_name=internal_name,
                vlen=self.mlen,
                preload_len=self.hbm_v_prefetch_amount,
            )

        var = VRAMMatrixVar(
            self,
            internal_name,
            input_var.shape,
            display_name=display_name,
            physical_shape=input_var.physical_shape,
        )
        self._tensors[internal_name] = var
        return var

    # ========================================================================
    # Store Operations
    # ========================================================================

    def store(
        self,
        tensor_var,
        name: str | None = None,
        hbm_addr: int | None = None,
        precision: int = 0,
        hbm_row_width: int | None = None,
        hbm_element_width: int | None = None,
        hbm_block_size: int | None = None,
        hbm_scale_width: int | None = None,
        precision_role: str | None = None,
    ) -> InputVar:
        """
        Write tensor from VRAM back to HBM.

        Returns:
            InputVar proxy object (can be loaded back later)
        """
        if not isinstance(tensor_var, VRAMMatrixVar):
            raise TypeError(f"Store requires VRAMMatrixVar, got {type(tensor_var)}")

        display_name = name if name is not None else f"{tensor_var.display_name}_stored"
        internal_name = self._scoped_name(display_name)
        if precision_role is None:
            precision_role = "key" if precision == 1 else "activation"

        if hbm_addr is None:
            h, w = tensor_var.physical_shape
            size = h * w
            hbm_size = self.hbm_tensor_size(
                size,
                hbm_row_width=hbm_row_width,
                hbm_element_width=hbm_element_width,
                hbm_block_size=hbm_block_size,
                hbm_scale_width=hbm_scale_width,
            )
            hbm_addr = self._allocate_hbm(hbm_size)
        else:
            h, w = tensor_var.physical_shape
            hbm_size = self.hbm_tensor_size(
                h * w,
                hbm_row_width=hbm_row_width,
                hbm_element_width=hbm_element_width,
                hbm_block_size=hbm_block_size,
                hbm_scale_width=hbm_scale_width,
            )

        super().store_to_hbm(
            tensor_name=tensor_var.name,  # internal name for symbol table lookup
            hbm_addr=hbm_addr,
            hbm_object_name=internal_name,
            vlen=self.mlen,
            precision=precision,
            store_amount=self.hbm_v_writeback_amount,
            hbm_row_width=hbm_row_width,
            hbm_element_width=hbm_element_width,
            hbm_block_size=hbm_block_size,
            hbm_scale_width=hbm_scale_width,
            precision_role=precision_role,
        )

        var = InputVar(
            self,
            internal_name,
            tensor_var.shape,
            hbm_addr,
            hbm_size,
            display_name=display_name,
            physical_shape=tensor_var.physical_shape,
        )
        self._inputs[internal_name] = var
        return var

    def append_packed_kv_row(
        self,
        tensor_var: VRAMMatrixVar,
        cache_var: InputVar,
        *,
        token_index: int,
        packed_layout: PackedKVLayout,
        role: str,
    ) -> PackedKVAppendAddress:
        """Append one logical cache row while preserving global MX planes."""

        if not isinstance(tensor_var, VRAMMatrixVar):
            raise TypeError("PackedKV append source must be a VRAMMatrixVar")
        if not isinstance(cache_var, InputVar):
            raise TypeError("PackedKV append destination must be an InputVar")
        if role not in {"key", "value"}:
            raise ValueError("PackedKV append role must be key or value")

        cache_layout = self.get_hbm_layout(cache_var.name)
        if cache_layout.precision_role != role:
            raise ValueError("PackedKV cache precision role differs")
        if tensor_var.shape[0] != 1:
            raise ValueError("PackedKV append source must have logical q_len=1")
        if tensor_var.shape[1] != packed_layout.active_elements:
            raise ValueError(
                "PackedKV append source width must equal kv_heads * head_dim"
            )
        if tensor_var.physical_shape[1] != packed_layout.mlen:
            raise ValueError("PackedKV append source must be MLEN-wide")
        transfer_rows = self.hbm_v_writeback_amount
        if tensor_var.physical_shape[0] < transfer_rows:
            raise ValueError(
                "PackedKV append source lacks H_STORE_V padding rows"
            )

        positions = getattr(self, "_packed_kv_append_positions", None)
        if positions is None:
            positions = {}
            self._packed_kv_append_positions = positions
        expected = positions.setdefault(cache_var.name, cache_var.shape[0])
        if token_index != expected:
            raise ValueError(
                f"PackedKV append expected token {expected}, got {token_index}"
            )
        address = resolve_packed_kv_append(
            cache_layout,
            packed_layout,
            token_index=token_index,
            transfer_rows=transfer_rows,
        )
        super().store_to_hbm(
            tensor_name=tensor_var.name,
            hbm_addr=cache_layout.hbm_base_addr,
            hbm_object_name=None,
            vlen=self.mlen,
            precision=1,
            store_amount=transfer_rows,
            hbm_row_width=cache_layout.hbm_row_width,
            hbm_element_width=cache_layout.hbm_element_width,
            hbm_block_size=cache_layout.hbm_block_size,
            hbm_scale_width=cache_layout.hbm_scale_width,
            precision_role=role,
            hbm_offset_bytes=address.element_offset_bytes,
            hbm_element_plane_bytes=address.element_plane_bytes,
            transfer_shape=(1, packed_layout.mlen),
            bind_tensor_hbm=False,
        )
        positions[cache_var.name] = token_index + 1
        return address

    def append_packed_kv_batch(
        self,
        tensor_var: VRAMMatrixVar,
        cache_var: InputVar,
        *,
        cache_position: int,
        batch_size: int,
        source_rows_per_batch: int,
        cache_rows_per_batch: int,
        packed_layout: PackedKVLayout,
        role: str,
    ) -> tuple[PackedKVAppendAddress, ...]:
        """Append one q_len=1 row to every independent cache slab."""

        if not isinstance(tensor_var, VRAMMatrixVar):
            raise TypeError("PackedKV append source must be a VRAMMatrixVar")
        if not isinstance(cache_var, InputVar):
            raise TypeError("PackedKV append destination must be an InputVar")
        if role not in {"key", "value"}:
            raise ValueError("PackedKV append role must be key or value")
        integer_values = (
            cache_position,
            batch_size,
            source_rows_per_batch,
            cache_rows_per_batch,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in integer_values):
            raise TypeError("PackedKV batch append coordinates must be integers")
        if cache_position < 0 or min(
            batch_size,
            source_rows_per_batch,
            cache_rows_per_batch,
        ) <= 0:
            raise ValueError("PackedKV batch append coordinates are invalid")
        if source_rows_per_batch % self.mlen:
            raise ValueError("PackedKV source rows per batch must be a multiple of MLEN")
        if cache_rows_per_batch % self.mlen:
            raise ValueError("PackedKV cache rows per batch must be a multiple of MLEN")
        if tensor_var.physical_shape != (
            batch_size * source_rows_per_batch,
            packed_layout.mlen,
        ):
            raise ValueError("PackedKV append source does not match the independent batch slabs")

        cache_layout = self.get_hbm_layout(cache_var.name)
        if cache_layout.precision_role != role:
            raise ValueError("PackedKV cache precision role differs")
        if cache_layout.physical_shape != (
            batch_size * cache_rows_per_batch,
            packed_layout.mlen,
        ):
            raise ValueError("PackedKV cache does not match the independent batch slabs")
        if cache_position + self.hbm_v_writeback_amount > cache_rows_per_batch:
            raise ValueError("PackedKV append transfer crosses a cache slab boundary")

        source_base = self.get_vram_addr(tensor_var.name)
        addresses = []
        for batch_index in range(batch_size):
            source_view = self.alloc_at(
                f"{tensor_var.display_name}_append_b{batch_index}",
                1,
                packed_layout.active_elements,
                source_base
                + batch_index * source_rows_per_batch * packed_layout.mlen,
                physical_shape=(source_rows_per_batch, packed_layout.mlen),
            )
            token_index = batch_index * cache_rows_per_batch + cache_position
            address = resolve_packed_kv_append(
                cache_layout,
                packed_layout,
                token_index=token_index,
                transfer_rows=self.hbm_v_writeback_amount,
            )
            super().store_to_hbm(
                tensor_name=source_view.name,
                hbm_addr=cache_layout.hbm_base_addr,
                hbm_object_name=None,
                vlen=self.mlen,
                precision=1,
                store_amount=self.hbm_v_writeback_amount,
                hbm_row_width=cache_layout.hbm_row_width,
                hbm_element_width=cache_layout.hbm_element_width,
                hbm_block_size=cache_layout.hbm_block_size,
                hbm_scale_width=cache_layout.hbm_scale_width,
                precision_role=role,
                hbm_offset_bytes=address.element_offset_bytes,
                hbm_element_plane_bytes=address.element_plane_bytes,
                transfer_shape=(1, packed_layout.mlen),
                bind_tensor_hbm=False,
            )
            self.free_tensor(source_view)
            addresses.append(address)
        return tuple(addresses)

    # ========================================================================
    # VRAM Matrix Allocation
    # ========================================================================

    def alloc(
        self,
        name: str,
        rows: int,
        cols: int,
        strict: bool = True,
        physical_shape: tuple[int, int] | None = None,
    ) -> VRAMMatrixVar:
        """
        Allocate a VRAM matrix.

        Used to store intermediate results (e.g., S block, PV, O).
        Within function scope, names are automatically prefixed to avoid conflicts.

        Args:
            name: matrix name (user-visible)
            rows: number of rows
            cols: number of columns
            strict: if False, skip mlen-alignment checks (for small scratch matrices)

        Returns:
            VRAMMatrixVar proxy object
        """
        display_name = name
        internal_name = self._scoped_name(name)
        if physical_shape is None and not strict:
            physical_rows = ((rows + self.blen - 1) // self.blen) * self.blen
            physical_cols = ((cols + self.mlen - 1) // self.mlen) * self.mlen
            physical_shape = (max(self.blen, physical_rows), max(self.mlen, physical_cols))
        super().allocate_vram_matrix(
            name=internal_name,
            rows=rows,
            cols=cols,
            strict=strict,
            physical_shape=physical_shape,
        )

        var = VRAMMatrixVar(
            self,
            internal_name,
            (rows, cols),
            display_name=display_name,
            physical_shape=physical_shape,
        )
        self._tensors[internal_name] = var
        return var

    def alloc_at(
        self,
        name: str,
        rows: int,
        cols: int,
        vram_addr: int,
        physical_shape: tuple[int, int] | None = None,
    ) -> VRAMMatrixVar:
        """Allocate a VRAM matrix view at a specific address.

        Used to create views into existing VRAM matrices (e.g., per-head
        slices of a multi-head Q projection output). Does NOT bump the
        VRAM allocator -- the caller is responsible for ensuring the region
        is valid.

        Args:
            name: matrix name (user-visible)
            rows: number of rows
            cols: number of columns
            vram_addr: absolute VRAM address for this view

        Returns:
            VRAMMatrixVar proxy object
        """
        display_name = name
        internal_name = self._scoped_name(name)
        self.add_vram_object(
            name=internal_name,
            shape=(rows, cols),
            physical_shape=physical_shape,
            vram_addr=vram_addr,
            allocate_if_none=False,
            strict=False,
        )
        isa_code = f"; VRAM View {name}: ({rows}, {cols}) at VRAM[{vram_addr}]\n"
        self.emit(isa_code)
        var = VRAMMatrixVar(
            self,
            internal_name,
            (rows, cols),
            display_name=display_name,
            physical_shape=physical_shape,
        )
        self._tensors[internal_name] = var
        return var

    def free_tensor(self, tensor_var: TensorVar):
        """
        Free a tensor in VRAM, reclaiming space for subsequent allocations.

        Freed space can be reused by new alloc() or other operations.
        """
        if not isinstance(tensor_var, VRAMMatrixVar):
            raise TypeError(f"Can only free VRAMMatrixVar, got {type(tensor_var)}")

        super().free_vram_object(tensor_var.name, strict=False)
        # Keep sub-matrix registration state consistent after free.
        self._registered_vram_sub_matrices[tensor_var.name] = False

    def free_input(self, input_var: InputVar):
        """
        Free an InputVar bookkeeping and recycle its HBM range for future auto-allocation.

        Notes:
        - This only affects PlenaCompiler's address management state.
        - If a freed input is referenced again later, caller is responsible for correctness.
        """
        if not isinstance(input_var, InputVar):
            raise TypeError(f"Can only free InputVar, got {type(input_var)}")

        super().free_hbm_object(input_var.name, strict=False)
        self._registered_hbm_sub_matrices[input_var.name] = False
        self._recycle_hbm(input_var.hbm_addr, input_var.hbm_size)
        self._inputs.pop(input_var.name, None)

    def free_fp_var(self, fp_var: FPVar):
        """
        Free an FPVar and return its block to FPRAM free pool.
        """
        if not isinstance(fp_var, FPVar):
            raise TypeError(f"Can only free FPVar, got {type(fp_var)}")
        self.free_fpram(fp_var.name, strict=True)

    # ========================================================================
    # Normalization Operations
    # ========================================================================

    def norm(
        self,
        tensor_var: TensorVar,
        mode: str = "rms",
        eps_offset: int = 1,
        reci_hid_offset: int = 2,
        vlen: int | None = None,
        scratchpad_vram_addr: int | None = None,
        destination_var: TensorVar | None = None,
    ) -> TensorVar:
        """
        Normalize a tensor, in-place or into a separate destination.

        Args:
            tensor_var: tensor to normalize (must have VRAM backing, e.g., VRAMMatrixVar)
            mode: "rms" or "layer"
            eps_offset: FPRAM address of epsilon
            reci_hid_offset: FPRAM address of 1/hidden_dim
            vlen: vector length (default: program mlen)
            scratchpad_vram_addr: optional scratchpad VRAM address
            destination_var: write the normalized rows here and leave the input
                intact, so the caller can keep it as a residual

        Returns:
            The tensor holding the normalized rows
        """
        if not isinstance(tensor_var, VRAMMatrixVar):
            raise TypeError(f"norm requires VRAMMatrixVar, got {type(tensor_var)}")
        if destination_var is not None and not isinstance(destination_var, VRAMMatrixVar):
            raise TypeError("norm destination must be a VRAMMatrixVar")

        super().normalize(
            tensor_name=tensor_var.name,
            mode=mode,
            eps_offset=eps_offset,
            reci_hid_offset=reci_hid_offset,
            vlen=vlen,
            scratchpad_vram_addr=scratchpad_vram_addr,
            destination_name=None if destination_var is None else destination_var.name,
        )
        return tensor_var if destination_var is None else destination_var

    def rms_norm(
        self,
        tensor_var: TensorVar,
        eps_offset: int = 1,
        reci_hid_offset: int = 2,
        vlen: int | None = None,
        scratchpad_vram_addr: int | None = None,
        destination_var: TensorVar | None = None,
    ) -> TensorVar:
        """RMS normalization, in-place unless *destination_var* is given."""
        return self.norm(
            tensor_var=tensor_var,
            mode="rms",
            eps_offset=eps_offset,
            reci_hid_offset=reci_hid_offset,
            vlen=vlen,
            scratchpad_vram_addr=scratchpad_vram_addr,
            destination_var=destination_var,
        )

    def layer_norm(
        self,
        tensor_var: TensorVar,
        eps_offset: int = 1,
        reci_hid_offset: int = 2,
        vlen: int | None = None,
        scratchpad_vram_addr: int | None = None,
    ) -> TensorVar:
        """Layer normalization (in-place)."""
        return self.norm(
            tensor_var=tensor_var,
            mode="layer",
            eps_offset=eps_offset,
            reci_hid_offset=reci_hid_offset,
            vlen=vlen,
            scratchpad_vram_addr=scratchpad_vram_addr,
        )

    def affine_rms_norm(
        self,
        tensor_var: TensorVar,
        weight_var: TensorVar,
        *,
        eps_offset: int = 3,
        reci_hid_offset: int = 4,
        vlen: int | None = None,
    ) -> TensorVar:
        """Apply RMSNorm and an expanded learned affine weight."""

        if not isinstance(tensor_var, VRAMMatrixVar):
            raise TypeError("affine_rms_norm requires a VRAM activation")
        if not isinstance(weight_var, VRAMMatrixVar):
            raise TypeError("affine_rms_norm requires a VRAM weight")
        if tensor_var.shape != weight_var.shape:
            raise ValueError("affine RMSNorm weight must match the logical shape")
        if tensor_var.physical_shape != weight_var.physical_shape:
            raise ValueError("affine RMSNorm weight must match physical storage")

        self.rms_norm(
            tensor_var,
            eps_offset=eps_offset,
            reci_hid_offset=reci_hid_offset,
            vlen=vlen,
        )
        self.vram_mul(tensor_var, weight_var, num_rows=tensor_var.shape[0])
        return tensor_var

    def segmented_affine_rms_norm(
        self,
        tensor_var: TensorVar,
        weight_var: TensorVar,
        *,
        segment_width: int,
        reci_segment_offset: int,
        eps_offset: int = 3,
        vlen: int | None = None,
    ) -> TensorVar:
        """Apply per-segment RMSNorm and a learned affine weight.

        A full weight tensor preserves the original elementwise path. A
        transfer-padded MLEN-wide pattern is reused across rows and column
        blocks, which represents shared per-head Q/K norm weights compactly.
        """

        if not isinstance(tensor_var, VRAMMatrixVar):
            raise TypeError("segmented RMSNorm requires a VRAM activation")
        if not isinstance(weight_var, VRAMMatrixVar):
            raise TypeError("segmented RMSNorm requires a VRAM weight")
        expanded_weight = (
            tensor_var.shape == weight_var.shape
            and tensor_var.physical_shape == weight_var.physical_shape
        )
        compact_pattern = (
            weight_var.shape[0] >= 1
            and weight_var.shape[1] == self.mlen
            and weight_var.physical_shape[0] >= 1
            and weight_var.physical_shape[1] == self.mlen
        )
        if not expanded_weight and not compact_pattern:
            raise ValueError(
                "segmented RMSNorm weight must match the activation or be "
                "an MLEN-wide broadcast pattern"
            )
        if compact_pattern and vlen not in (None, self.mlen):
            raise ValueError("compact segmented RMSNorm weights require VLEN=MLEN")

        super().segmented_rms_normalize(
            tensor_var.name,
            segment_width=segment_width,
            eps_offset=eps_offset,
            reci_segment_offset=reci_segment_offset,
            vlen=vlen,
        )
        if expanded_weight:
            self.vram_mul(tensor_var, weight_var, num_rows=tensor_var.shape[0])
        else:
            self.vram_broadcast_row_mul(
                tensor_var,
                weight_var,
                num_rows=tensor_var.shape[0],
            )
        return tensor_var

    def silu(
        self,
        tensor_var: TensorVar,
        *,
        const_one_fp_address: int = 5,
        vlen: int | None = None,
    ) -> TensorVar:
        """Apply SiLU in-place."""

        if not isinstance(tensor_var, VRAMMatrixVar):
            raise TypeError("silu requires a VRAM activation")
        super().silu(
            tensor_var.name,
            const_one_fp_address=const_one_fp_address,
            vlen=vlen,
        )
        return tensor_var

    # ========================================================================
    # Composite Decoder Operations
    # ========================================================================

    def ffn(self, input_var: VRAMMatrixVar, w_gate: InputVar, w_up: InputVar, w_down: InputVar):
        """Emit the fused FFN kernel and return the in-place activation var."""
        batch_size, hidden_size = input_var.physical_shape
        _, inter_dim = w_up.physical_shape
        mlen = self.mlen
        blen = self.blen
        # rows//blen drives the inner activation-column loop; a non-multiple
        # (esp. rows < blen) emits C_LOOP_START 0 and the emulator panics.
        if batch_size <= 0 or batch_size % blen != 0:
            raise ValueError(
                f"FFN activation rows ({batch_size}) must be a positive multiple of BLEN ({blen})."
            )
        activation_base_address = self.get_vram_addr(input_var.name)
        max_k_tiles = max(hidden_size // mlen, inter_dim // mlen)
        use_loop_instructions = max_k_tiles <= self.mram_tile_capacity
        workspace_elems = batch_size * (2 * inter_dim + max(hidden_size, inter_dim))
        workspace_rows = (workspace_elems + mlen - 1) // mlen
        workspace = self.alloc(
            "_ffn_workspace",
            workspace_rows,
            mlen,
            strict=False,
            physical_shape=(workspace_rows, mlen),
        )
        workspace_base_address = self.get_vram_addr(workspace.name)
        gate_layout = self.get_hbm_layout(w_gate.name)
        up_layout = self.get_hbm_layout(w_up.name)
        down_layout = self.get_hbm_layout(w_down.name)
        weight_widths = {
            layout.hbm_element_width
            for layout in (gate_layout, up_layout, down_layout)
        }
        if len(weight_widths) != 1:
            raise ValueError("fused FFN weights must use one element width")
        if (
            gate_layout.element_plane_bytes != up_layout.element_plane_bytes
            or gate_layout.element_stride_bytes(inter_dim)
            != up_layout.element_stride_bytes(inter_dim)
        ):
            raise ValueError("fused FFN gate and up layouts must match")

        isa_code = preload_addr_reg_asm(
            addr_reg_to_set=[1, 2, 3],
            available_registers=[1, 2, 3],
            addr_reg_val=[w_gate.hbm_addr, w_up.hbm_addr, w_down.hbm_addr],
        )
        isa_code += reset_reg_asm(alive_registers=[1, 2, 3])
        isa_code += ffn_asm(
            mlen=mlen,
            vlen=mlen,
            blen=blen,
            batch=batch_size,
            seq_len=1,
            hidden_size=hidden_size,
            intermediate_size=inter_dim,
            alive_registers=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            gate_weight_hbm_offset_reg=1,
            up_weight_hbm_offset_reg=2,
            down_weight_hbm_offset_reg=3,
            const_one_fp_address=5,
            activation_base_address=activation_base_address,
            use_loop_instructions=use_loop_instructions,
            matrix_sram_size=self.mram_capacity_elems,
            workspace_base_address=workspace_base_address,
            weight_element_bits=up_layout.hbm_element_width,
            up_weight_element_plane_bytes=up_layout.element_plane_bytes,
            up_weight_stride_bytes=up_layout.element_stride_bytes(inter_dim),
            down_weight_element_plane_bytes=down_layout.element_plane_bytes,
            down_weight_stride_bytes=down_layout.element_stride_bytes(hidden_size),
        )

        self.emit(isa_code)
        self.free_tensor(workspace)
        return input_var


__all__ = ["ProgramTensorMixin"]
