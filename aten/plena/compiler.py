"""User-facing PLENA compiler DSL."""

from __future__ import annotations

import os
from collections.abc import Callable
from functools import wraps

from compiler.aten.plena.isa_compiler import IsaCompiler
from compiler.aten.plena.dsl_attention import DslAttentionMixin
from compiler.aten.plena.dsl_fp_tile_ops import DslFPTileOpsMixin
from compiler.aten.plena.dsl_matrix_ops import DslMatrixOpsMixin
from compiler.aten.plena.dsl_tensors import DslTensorMixin
from compiler.aten.plena.vars import FPVar, InputVar, TensorVar, VRAMMatrixVar


class _IsaCompilerView:
    """
    Back-compat proxy for legacy ``prog._compiler.X(...)`` call sites.

    PlenaCompiler now inherits IsaCompiler rather than composing it, so
    for call sites that still expect to reach the low-level IsaCompiler
    API (e.g., ``allocate_fpram(name=..., size=...)`` returning an int), we
    expose a proxy whose attribute lookup resolves callables on
    IsaCompiler directly (bypassing any PlenaCompiler overrides with
    colliding names). Non-callable attributes (e.g., ``generated_code``,
    ``vram_allocator``) fall through to the underlying instance unchanged.
    """

    __slots__ = ("_inst",)

    def __init__(self, inst: PlenaCompiler):
        object.__setattr__(self, "_inst", inst)

    def __getattr__(self, name: str):
        cls_attr = getattr(IsaCompiler, name, None)
        if cls_attr is not None and callable(cls_attr):
            return cls_attr.__get__(self._inst, IsaCompiler)
        return getattr(self._inst, name)

    def __setattr__(self, name: str, value):
        setattr(self._inst, name, value)


# ============================================================================
# PlenaCompiler Main Class
# ============================================================================


class PlenaCompiler(
    DslTensorMixin,
    DslFPTileOpsMixin,
    DslMatrixOpsMixin,
    DslAttentionMixin,
    IsaCompiler,
):
    """
    PLENA High-level Compiler Interface.

    Inherits the ISA-emission machinery from IsaCompiler and layers a
    Pythonic DSL on top. All operations are eagerly evaluated — ISA code is
    generated immediately upon call.
    """

    def __init__(self, mlen: int = 64, blen: int = 4, real_data_ratio: float = 1.125, unroll_loops: bool = False):
        """
        Args:
            mlen: Matrix tile size (default 64)
            blen: Vector tile size (default 4)
            real_data_ratio: HBM storage ratio (MXFP8 format = 1.125)
            unroll_loops: If True, unroll sub-projection loops at ASM-gen time to
                          eliminate C_LOOP_START/END overhead. Overridden by the
                          ATEN_UNROLL env var ("1"=True, "0"=False).
        """
        _env_unroll = os.environ.get("ATEN_UNROLL", "")
        if _env_unroll == "1":
            unroll_loops = True
        elif _env_unroll == "0":
            unroll_loops = False
        super().__init__(mlen=mlen, blen=blen, real_data_ratio=real_data_ratio, unroll_loops=unroll_loops)

        # HBM address auto-allocation
        self._next_hbm_addr: int = 0
        self._hbm_free_blocks: list[tuple[int, int]] = []  # (addr, size)

        # Variable registries
        self._inputs: dict[str, InputVar] = {}
        self._tensors: dict[str, TensorVar] = {}
        self._fp_vars: dict[str, FPVar] = {}
        self._functions: dict[str, Callable] = {}
        self._registered_hbm_sub_matrices: dict[str, bool] = {}
        self._registered_vram_sub_matrices: dict[str, bool] = {}

        self._result_tensor: TensorVar | None = None

        # Auto-generated name counter
        self._auto_name_counter: int = 0

        # Function scope namespace
        # Push a prefix on each function call (e.g., "linear_0/"), pop on exit
        # _auto_name will automatically add current scope prefix, avoiding name conflicts when calling the same function multiple times
        self._scope_stack: list[str] = []
        self._function_call_counters: dict[str, int] = {}  # func_name -> call count

    # ========================================================================
    # Property Access
    # ========================================================================

    # mlen / blen are instance attributes inherited from TileCompiler.__init__.

    @property
    def compiler(self) -> PlenaCompiler:
        """Legacy accessor — returns self now that PlenaCompiler is the compiler."""
        return self

    @property
    def _compiler(self) -> _IsaCompilerView:
        """Back-compat shim for legacy ``prog._compiler.X(...)`` call sites.
        Returns a proxy that resolves callables against IsaCompiler
        directly so callers reach the low-level API regardless of any
        PlenaCompiler override with the same name."""
        return _IsaCompilerView(self)

    @property
    def symbol_table(self):
        """Access symbol table."""
        return self.get_symbol_table()

    # ========================================================================
    # Function Decorator
    # ========================================================================

    def function(self, func: Callable) -> Callable:
        """
        Decorator: Define reusable functions.

        Each invocation generates fresh ISA code (eager evaluation).
        Internally allocated tensors are auto-freed on exit unless returned.

        Scoping: intermediate tensors get a call-index prefix to avoid name
        collisions across repeated calls (e.g., "linear_0/proj_1", "linear_1/proj_1").
        Nested functions compose prefixes: "two_layer_0/linear_0/proj_1".
        """
        func_name = func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            call_idx = self._function_call_counters.get(func_name, 0)
            self._function_call_counters[func_name] = call_idx + 1

            scope = f"{func_name}_{call_idx}/"
            self._scope_stack.append(scope)

            self.emit_comment(f"=== Enter {func_name} (call #{call_idx}) ===")

            # Snapshot: record existing tensors before function execution
            tensors_before = set(self._tensors.keys())
            inputs_before = set(self._inputs.keys())
            fp_vars_before = set(self._fp_vars.keys())

            try:
                result = func(*args, **kwargs)

                # Auto-free: free locally allocated tensors that are not returned
                return_names = set()
                return_fp_names = set()
                if isinstance(result, TensorVar):
                    return_names.add(result.internal_name)
                elif isinstance(result, FPVar):
                    return_fp_names.add(result.internal_name)
                elif isinstance(result, (tuple, list)):
                    for r in result:
                        if isinstance(r, TensorVar):
                            return_names.add(r.internal_name)
                        elif isinstance(r, FPVar):
                            return_fp_names.add(r.internal_name)

                for name in set(self._tensors.keys()) - tensors_before:
                    if name not in return_names:
                        tensor = self._tensors[name]
                        if isinstance(tensor, VRAMMatrixVar):
                            self.free_tensor(tensor)
                            self._registered_vram_sub_matrices[tensor.name] = False

                for name in set(self._inputs.keys()) - inputs_before:
                    if name not in return_names:
                        self.free_input(self._inputs[name])

                local_fp_names = sorted(
                    set(self._fp_vars.keys()) - fp_vars_before,
                    key=lambda n: self._fp_vars[n].address,
                    reverse=True,
                )
                for name in local_fp_names:
                    if name in return_fp_names:
                        continue
                    fp_var = self._fp_vars.get(name)
                    if fp_var is not None:
                        self.free_fp_var(fp_var)
            finally:
                self._scope_stack.pop()
                self.emit_comment(f"=== Exit {func_name} (call #{call_idx}) ===")

            return result

        self._functions[func_name] = wrapper
        wrapper._plena_function = True
        wrapper._plena_name = func_name
        return wrapper

    # ========================================================================
    # Result Marking
    # ========================================================================

    def result(self, tensor_var: TensorVar):
        """Mark output result tensor."""
        self._result_tensor = tensor_var

    # ========================================================================
    # Compilation
    # ========================================================================

    def compile(self) -> str:
        """Get generated ISA code string."""
        return super().get_code()

    def print_symbol_table(self):
        """Print symbol table"""
        super().print_symbol_table()

    def get_symbol_table(self):
        """Get symbol table"""
        return super().get_symbol_table()

    # ========================================================================
    # Operator Dispatch (internal)
    # ========================================================================

    def _dispatch_matmul(self, left: TensorVar, right) -> TensorVar:
        raise TypeError("@ operator is no longer supported in PlenaCompiler. Use explicit program APIs instead.")

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def _scoped_name(self, name: str) -> str:
        """
        Apply current scope prefix to a name.

        - Top-level alloc("temp"):                    -> "temp"
        - Inside linear call 0, alloc("temp"):        -> "linear_0/temp"
        - Nested two_layer->linear, alloc("temp"):    -> "two_layer_0/linear_0/temp"
        """
        if not self._scope_stack:
            return name
        scope_prefix = "".join(self._scope_stack)
        return f"{scope_prefix}{name}"

    def _allocate_hbm(self, hbm_size: int) -> int:
        """Allocate HBM range, preferring previously freed blocks."""
        best_idx = None
        best_waste = None
        for i, (addr, size) in enumerate(self._hbm_free_blocks):
            if size >= hbm_size:
                waste = size - hbm_size
                if best_waste is None or waste < best_waste:
                    best_idx = i
                    best_waste = waste

        if best_idx is not None:
            addr, block_size = self._hbm_free_blocks.pop(best_idx)
            # Return excess fragment to free list
            excess = block_size - hbm_size
            if excess > 0:
                self._hbm_free_blocks.append((addr + hbm_size, excess))
            return addr

        addr = self._next_hbm_addr
        m = self.mlen
        self._next_hbm_addr = ((addr + hbm_size + m - 1) // m) * m
        return addr

    def _recycle_hbm(self, hbm_addr: int, hbm_size: int):
        """Recycle an HBM range for future auto-allocation."""
        if hbm_size <= 0:
            return
        self._hbm_free_blocks.append((hbm_addr, hbm_size))

    def _auto_name(self, prefix: str = "t") -> str:
        """
        Generate a unique scoped name.

        - Top-level:          "__proj_1"
        - linear call 0:      "linear_0/__proj_1"
        - nested:             "two_layer_0/linear_0/__proj_1"
        """
        self._auto_name_counter += 1
        scope_prefix = "".join(self._scope_stack)
        return f"{scope_prefix}__{prefix}_{self._auto_name_counter}"

    def __repr__(self):
        num_inputs = len(self._inputs)
        num_tensors = len(self._tensors)
        num_functions = len(self._functions)
        code_len = len(super().get_code().splitlines())
        return (
            f"PlenaCompiler(mlen={self.mlen}, blen={self.blen}, "
            f"inputs={num_inputs}, tensors={num_tensors}, "
            f"functions={num_functions}, isa_lines={code_len})"
        )


__all__ = ["PlenaCompiler"]
