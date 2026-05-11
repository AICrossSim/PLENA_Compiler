"""Tensor, memory, and normalization operations for the PLENA DSL."""

from __future__ import annotations

from compiler.aten.plena.vars import FPVar, InputVar, TensorVar, VRAMMatrixVar


class DslTensorMixin:
    # ========================================================================
    # Input Declaration
    # ========================================================================

    def input(
        self,
        name: str,
        shape: tuple[int, int],
        hbm_addr: int | None = None,
        prestaged_vram_addr: int | None = None,
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
        h, w = shape
        size = h * w
        hbm_size = int(size * self.real_data_ratio)

        if hbm_addr is None:
            hbm_addr = self._allocate_hbm(hbm_size)

        var = InputVar(self, name, shape, hbm_addr, hbm_size, prestaged_vram_addr=prestaged_vram_addr)
        self._inputs[name] = var
        super().add_hbm_object(
            name=name,
            hbm_addr=hbm_addr,
            shape=shape,
            real_data_ratio=self.real_data_ratio,
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
            h, w = input_var.shape
            vram_addr = input_var.prestaged_vram_addr
            # Tell the VRAM allocator that this region is occupied so subsequent
            # allocations don't collide with it.
            self.vram_allocator._vmm.mark_used(vram_addr, h * w, name=internal_name)
            super().add_vram_object(
                name=internal_name,
                shape=(h, w),
                vram_addr=vram_addr,
                dtype="fp16",
                kind="Batch",
                allocate_if_none=False,
                strict=False,
            )
        else:
            # Normal path: emit HBM → VRAM prefetch ISA.
            super().load_batch(
                hbm_object_name=input_var.name, vram_object_name=internal_name, vlen=self.mlen, preload_len=4
            )

        var = VRAMMatrixVar(self, internal_name, input_var.shape, display_name=display_name)
        self._tensors[internal_name] = var
        return var

    # ========================================================================
    # Store Operations
    # ========================================================================

    def store(self, tensor_var, name: str | None = None, hbm_addr: int | None = None) -> InputVar:
        """
        Write tensor from VRAM back to HBM.

        Returns:
            InputVar proxy object (can be loaded back later)
        """
        if not isinstance(tensor_var, VRAMMatrixVar):
            raise TypeError(f"Store requires VRAMMatrixVar, got {type(tensor_var)}")

        display_name = name if name is not None else f"{tensor_var.display_name}_stored"
        internal_name = self._scoped_name(display_name)

        if hbm_addr is None:
            h, w = tensor_var.shape
            size = h * w
            hbm_size = int(size * self.real_data_ratio)
            hbm_addr = self._allocate_hbm(hbm_size)
        else:
            h, w = tensor_var.shape
            hbm_size = int(h * w * self.real_data_ratio)

        super().store_to_hbm(
            tensor_name=tensor_var.name,  # internal name for symbol table lookup
            hbm_addr=hbm_addr,
            hbm_object_name=internal_name,
            vlen=self.mlen,
        )

        var = InputVar(self, internal_name, tensor_var.shape, hbm_addr, hbm_size, display_name=display_name)
        self._inputs[internal_name] = var
        return var

    # ========================================================================
    # VRAM Matrix Allocation
    # ========================================================================

    def alloc(self, name: str, rows: int, cols: int, strict: bool = True) -> VRAMMatrixVar:
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
        super().allocate_vram_matrix(name=internal_name, rows=rows, cols=cols, strict=strict)

        var = VRAMMatrixVar(self, internal_name, (rows, cols), display_name=display_name)
        self._tensors[internal_name] = var
        return var

    def alloc_at(self, name: str, rows: int, cols: int, vram_addr: int) -> VRAMMatrixVar:
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
        self._compiler.add_vram_object(
            name=internal_name,
            shape=(rows, cols),
            vram_addr=vram_addr,
            allocate_if_none=False,
        )
        isa_code = f"; VRAM View {name}: ({rows}, {cols}) at VRAM[{vram_addr}]\n"
        self._compiler.emit(isa_code)
        var = VRAMMatrixVar(self, internal_name, (rows, cols), display_name=display_name)
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
    ) -> TensorVar:
        """
        Normalize tensor in-place.

        Args:
            tensor_var: tensor to normalize (must have VRAM backing, e.g., VRAMMatrixVar)
            mode: "rms" or "layer"
            eps_offset: FPRAM address of epsilon
            reci_hid_offset: FPRAM address of 1/hidden_dim
            vlen: vector length (default: program mlen)
            scratchpad_vram_addr: optional scratchpad VRAM address

        Returns:
            The same tensor_var (in-place operation)
        """
        if not isinstance(tensor_var, VRAMMatrixVar):
            raise TypeError(f"norm requires VRAMMatrixVar, got {type(tensor_var)}")

        super().normalize(
            tensor_name=tensor_var.name,
            mode=mode,
            eps_offset=eps_offset,
            reci_hid_offset=reci_hid_offset,
            vlen=vlen,
            scratchpad_vram_addr=scratchpad_vram_addr,
        )
        return tensor_var

    def rms_norm(
        self,
        tensor_var: TensorVar,
        eps_offset: int = 1,
        reci_hid_offset: int = 2,
        vlen: int | None = None,
        scratchpad_vram_addr: int | None = None,
    ) -> TensorVar:
        """RMS normalization (in-place)."""
        return self.norm(
            tensor_var=tensor_var,
            mode="rms",
            eps_offset=eps_offset,
            reci_hid_offset=reci_hid_offset,
            vlen=vlen,
            scratchpad_vram_addr=scratchpad_vram_addr,
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


__all__ = ["DslTensorMixin"]
