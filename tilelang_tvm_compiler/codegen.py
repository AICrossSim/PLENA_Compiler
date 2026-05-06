"""TIR -> PLENA pseudo-ISA codegen.

Walks a `tvm.tir.PrimFunc`:

  1. Collects every buffer (params + alloc_buffer inside Blocks) and its scope.
  2. Walks the body and finds `T.call_extern("handle", "plena.*", ...)` sites.
  3. Looks up the intrinsic spec, type-checks operand scopes, emits ISA text.

This is the equivalent of an MLIR "convert-plena-to-isa" pass, written
imperatively in Python because we are using TVM (no dialect machinery).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import tvm
from tvm import tir
from tvm.tir import stmt_functor

from . import intrinsics as _intrin
from . import scope as _scope
from . import hlir as _hlir


class CodegenError(RuntimeError):
    pass


class _BufferInfo:
    """What we remember per buffer for ISA emission."""

    __slots__ = ("name", "scope", "shape", "dtype")

    def __init__(self, name: str, scope: str, shape, dtype: str):
        self.name = name
        self.scope = scope
        self.shape = tuple(int(s) if isinstance(s, (int, tir.IntImm)) else s for s in shape)
        self.dtype = dtype

    def __repr__(self) -> str:
        return f"{self.name}<{self.scope}>"


def _normalize_scope(s: str) -> str:
    """Map TVM's default empty/"global" scope to our HBM."""
    if s in ("", "global"):
        return _scope.HBM
    return s


class PlenaCodegen:
    """One instance per PrimFunc compile."""

    def __init__(self, func: tir.PrimFunc, name: str = "kernel"):
        self.func = func
        self.name = name
        # data-handle Var -> _BufferInfo
        self._buffers: Dict[tir.Var, _BufferInfo] = {}
        # name-keyed lookup for diagnostic messages
        self._buffers_by_name: Dict[str, _BufferInfo] = {}
        self._isa_lines: List[str] = []

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def buffers_by_name(self) -> Dict[str, "_BufferInfo"]:
        """Read-only view of {buffer_name -> info}. Populated after run()."""
        return dict(self._buffers_by_name)

    def lower_to_hlir(self) -> _hlir.HLIRModule:
        """Pass 1: walk TIR -> HLIR module (buffers + ordered op stream).

        Replaces the text-based `run()` for the new pipeline. We keep
        both paths because `run()` is convenient for quick eyeballing
        of a kernel during development.
        """
        # Reuse the buffer collection logic from run().
        self._buffers.clear()
        self._buffers_by_name.clear()
        self._collect_param_buffers()
        self._collect_alloc_buffers()

        # Construct HLIR buffers (preserving param order).
        hlir_buffers: Dict[str, _hlir.Buffer] = {}
        param_names: List[str] = []
        for var in self.func.params:
            buf = self.func.buffer_map.get(var, None)
            if buf is None:
                continue
            info = self._buffers_by_name[buf.name]
            hlir_buffers[info.name] = self._buf_info_to_hlir(info)
            param_names.append(info.name)
        for name, info in self._buffers_by_name.items():
            if name not in hlir_buffers:
                hlir_buffers[name] = self._buf_info_to_hlir(info)

        # Walk the body and collect Op stream.
        ops: List[_hlir.Op] = []
        self._collect_ops(self.func.body, ops)

        return _hlir.HLIRModule(
            name=self.name,
            buffers=hlir_buffers,
            ops=ops,
            param_names=param_names,
        )

    @staticmethod
    def _buf_info_to_hlir(info: "_BufferInfo") -> _hlir.Buffer:
        return _hlir.Buffer(
            name=info.name,
            scope=info.scope,
            shape=tuple(int(s) for s in info.shape),
            dtype=info.dtype,
        )

    def _collect_ops(self, stmt, ops: List[_hlir.Op]) -> None:
        if isinstance(stmt, tir.SeqStmt):
            for s in stmt:
                self._collect_ops(s, ops)
        elif isinstance(stmt, tir.BlockRealize):
            self._collect_ops(stmt.block, ops)
        elif isinstance(stmt, tir.Block):
            self._collect_ops(stmt.body, ops)
        elif isinstance(stmt, tir.LetStmt):
            self._collect_ops(stmt.body, ops)
        elif isinstance(stmt, tir.IfThenElse):
            # PLENA's ISA has no scalar branch instructions, and the previous
            # "literal True/False only" handling was misleading -- it covered
            # essentially nothing that a Python-level `if` at kernel-build
            # time can't already express more clearly. Reject all TIR ifs.
            raise CodegenError(
                "tir.IfThenElse is not supported. Use a Python-level `if` in "
                "the kernel factory to specialize at build time, or T.unroll "
                "+ Python branching for per-iteration variants. PLENA has no "
                "branch ISA, so dynamic conditions cannot be lowered."
            )
        elif isinstance(stmt, tir.For):
            # Recursively collect the body into a fresh op list, then wrap
            # it in a structured ForOp. Pass 3 walks `body` while binding
            # `loop_var` to a GP register so PrimExprs that reference it
            # can be materialised by ExprMaterializer.
            body_ops: List[_hlir.Op] = []
            self._collect_ops(stmt.body, body_ops)
            extent = (
                int(stmt.extent.value) if isinstance(stmt.extent, tir.IntImm)
                else stmt.extent
            )
            init = (
                int(stmt.min.value) if isinstance(stmt.min, tir.IntImm)
                else stmt.min
            )
            # Capture loop kind so isa_pass can pick between hardware-loop
            # emission (C_LOOP_START/END) and full unrolling. T.unroll(...)
            # in the kernel maps to ForKind.UNROLLED here; isa_pass uses
            # this to escape the emulator's MAX_LOOP_INSTRUCTIONS-per-iter
            # cap when one iteration of an outer loop dispatches a body
            # too large to fit (e.g. the 16x16 emit_matmul expansion).
            # tir.For.kind is an int-valued enum member; resolve to a name.
            raw_kind = getattr(stmt, "kind", None)
            try:
                kind_str = tir.ForKind(int(raw_kind)).name.lower()
            except (TypeError, ValueError):
                kind_str = "serial"
            for_op = _hlir.make_for_op(
                loop_var=stmt.loop_var,
                extent=extent,
                body=body_ops,
                init=init,
            )
            for_op.annotations["loop_kind"] = kind_str
            ops.append(for_op)
        elif isinstance(stmt, tir.Evaluate):
            self._collect_op_from_evaluate(stmt, ops)
        elif isinstance(stmt, tir.AttrStmt):
            self._collect_ops(stmt.body, ops)

    def _collect_slice_op(
        self,
        val: tir.Call,
        name: str,
        kind: str,
        ops: List[_hlir.Op],
    ) -> None:
        """Parse `plena.dma_*_slice` calls.

        Layout:
            args[1]   src_buf.data       (Var)
            args[2]   dst_buf.data       (Var)
            args[3]   ndim               (IntImm)
            args[4..4+ndim-1]            starts (PrimExpr / IntImm)
            args[4+ndim..4+2*ndim-1]     extents (IntImm)

        The src OR dst is the sliced one, depending on direction:
            h2v / h2m  -> src is sliced (HBM tensor)
            v2h        -> dst is sliced (writing to a sub-region of HBM)
        """
        raw = list(val.args[1:])
        if len(raw) < 4:
            raise CodegenError(
                f"{name}: expected at least 4 args (src, dst, ndim, ...), got {len(raw)}"
            )
        src_var, dst_var, ndim_imm = raw[0], raw[1], raw[2]
        if not isinstance(ndim_imm, tir.IntImm):
            raise CodegenError(
                f"{name}: ndim must be a compile-time int, got {type(ndim_imm).__name__}"
            )
        ndim = int(ndim_imm.value)
        if len(raw) != 3 + 2 * ndim:
            raise CodegenError(
                f"{name}: with ndim={ndim} expected exactly {3 + 2 * ndim} args, "
                f"got {len(raw)}"
            )
        starts_raw = raw[3 : 3 + ndim]
        extents_raw = raw[3 + ndim : 3 + 2 * ndim]

        # Each start may be int / IntImm (static) or arbitrary PrimExpr
        # (dynamic). Pass 3 will dispatch on type.
        starts: List[Any] = []
        for s in starts_raw:
            if isinstance(s, tir.IntImm):
                starts.append(int(s.value))
            elif isinstance(s, tir.PrimExpr):
                starts.append(s)
            else:
                raise CodegenError(
                    f"{name}: start must be IntImm or PrimExpr, got {type(s).__name__}"
                )
        extents: List[int] = []
        for e in extents_raw:
            if not isinstance(e, tir.IntImm):
                raise CodegenError(
                    f"{name}: extent must be a compile-time int, got "
                    f"{type(e).__name__}={e!r}"
                )
            extents.append(int(e.value))

        # Look up parent buffers from the data-handle Vars.
        if not (isinstance(src_var, tir.Var) and src_var in self._buffers):
            raise CodegenError(f"{name}: src is not a known buffer handle")
        if not (isinstance(dst_var, tir.Var) and dst_var in self._buffers):
            raise CodegenError(f"{name}: dst is not a known buffer handle")
        src_info = self._buffers[src_var]
        dst_info = self._buffers[dst_var]

        # Decide which side is sliced based on the intrinsic.
        if name in ("plena.dma_h2v_slice", "plena.dma_h2m_slice"):
            sliced = _hlir.BufferSlice(
                parent=src_info.name, starts=tuple(starts), extents=tuple(extents),
            )
            buffer_args: List[Any] = [sliced, dst_info.name]
        elif name == "plena.dma_v2h_slice":
            sliced = _hlir.BufferSlice(
                parent=dst_info.name, starts=tuple(starts), extents=tuple(extents),
            )
            buffer_args = [src_info.name, sliced]
        else:
            raise CodegenError(f"unhandled slice intrinsic: {name}")

        ops.append(_hlir.Op(
            kind=kind,
            buffer_args=buffer_args,
            scalar_args=[],
            annotations={"intrinsic": name},
        ))

    def _collect_op_from_evaluate(self, ev: tir.Evaluate, ops: List[_hlir.Op]) -> None:
        val = ev.value
        if not isinstance(val, tir.Call):
            return
        name = self._call_extern_name(val)
        if name is None or not name.startswith("plena."):
            return
        spec = _intrin.lookup(name)  # validates that the op is known
        kind = name[len("plena."):]

        # Slice variants have a structured arg pack: src, dst, ndim,
        # *starts, *extents. Pack the variadic suffix into a BufferSlice
        # and produce an HLIR Op whose `buffer_args[0]` is the slice (or
        # for v2h_slice, `buffer_args[1]`).
        if name.endswith("_slice"):
            self._collect_slice_op(val, name, kind, ops)
            return

        # Arg resolution. Buffer-handle Vars (those that map to a Buffer
        # we've already collected) become buffer_args by name. Everything
        # else is a scalar argument:
        #   - IntImm / FloatImm / StringImm -> native Python int/float/str
        #     (cheaper for downstream passes than carrying the IR node)
        #   - any other PrimExpr (loop var, compound expression like
        #     kv_block * mlen + offset) -> kept as-is so ExprMaterializer
        #     can lower it at ISA emit time
        raw_args = list(val.args[1:])
        buffer_args: List[str] = []
        scalar_args: List[Any] = []
        scopes: List[Optional[str]] = []
        for a in raw_args:
            if isinstance(a, tir.Var) and a in self._buffers:
                info = self._buffers[a]
                buffer_args.append(info.name)
                scopes.append(info.scope)
                continue
            if isinstance(a, tir.BufferLoad) and a.buffer.data in self._buffers:
                info = self._buffers[a.buffer.data]
                if info.scope == _scope.FPRAM:
                    scalar_args.append(_hlir.BufferElement(
                        buffer=info.name,
                        indices=tuple(self._normalize_scalar_expr(i) for i in a.indices),
                    ))
                    scopes.append(None)
                    continue
            scopes.append(None)
            scalar_args.append(self._normalize_scalar_expr(a))
        # Verify scopes against the registered intrinsic spec. We collapse
        # scopes from buffer/scalar args back into the original positional
        # order so verification matches op signatures.
        ordered_scopes: List[Optional[str]] = []
        bi = 0
        si = 0
        for a in raw_args:
            if isinstance(a, tir.Var) and a in self._buffers:
                ordered_scopes.append(self._buffers[a].scope)
                bi += 1
            else:
                ordered_scopes.append(None)
                si += 1
        self._verify_scopes(spec, name, ordered_scopes)

        ops.append(_hlir.Op(
            kind=kind,
            buffer_args=buffer_args,
            scalar_args=scalar_args,
            annotations={"intrinsic": name},
        ))

    def run(self) -> str:
        self._collect_param_buffers()
        self._collect_alloc_buffers()
        self._emit_header()
        self._emit_buffer_directives()
        self._isa_lines.append("")
        self._emit_body()
        return "\n".join(self._isa_lines) + "\n"

    # ------------------------------------------------------------------
    # buffer collection
    # ------------------------------------------------------------------
    def _collect_param_buffers(self) -> None:
        for var in self.func.params:
            buf = self.func.buffer_map.get(var, None)
            if buf is None:
                # opaque handle / scalar param -- skip for now
                continue
            self._record_buffer(buf, default_scope=_scope.HBM)

    def _collect_alloc_buffers(self) -> None:
        def visitor(node):
            if isinstance(node, tir.Block):
                for buf in node.alloc_buffers:
                    self._record_buffer(buf, default_scope=_scope.HBM)
            elif isinstance(node, tir.Allocate):
                # post-block-flattening form -- not used in our entry IR but
                # cheap to support so that lowering passes don't break us.
                pass

        stmt_functor.post_order_visit(self.func.body, visitor)

    def _record_buffer(self, buf: tir.Buffer, default_scope: str) -> None:
        scope = _normalize_scope(buf.scope() or default_scope)
        if not _scope.is_known(scope):
            raise CodegenError(
                f"buffer {buf.name!r} has unknown scope {scope!r}; "
                f"expected one of {_scope.ALL_SCOPES}"
            )
        info = _BufferInfo(buf.name, scope, buf.shape, str(buf.dtype))
        self._buffers[buf.data] = info
        self._buffers_by_name[buf.name] = info

    # ------------------------------------------------------------------
    # body walk
    # ------------------------------------------------------------------
    def _emit_body(self) -> None:
        # Use a manual recursive walk so we can preserve emission order.
        # post_order_visit reverses statements, which would scramble the ISA.
        self._walk_stmt(self.func.body)

    def _walk_stmt(self, stmt) -> None:
        if isinstance(stmt, tir.SeqStmt):
            for s in stmt:
                self._walk_stmt(s)
        elif isinstance(stmt, tir.BlockRealize):
            self._walk_stmt(stmt.block)
        elif isinstance(stmt, tir.Block):
            self._walk_stmt(stmt.body)
        elif isinstance(stmt, tir.For):
            # We don't emit loop control yet -- just unroll-by-walking.
            # Real PLENA would lower this to C_LOOP_START/END. For the
            # skeleton kernel there are no loops.
            self._isa_lines.append(
                f"; for {stmt.loop_var.name_hint} in [{stmt.min}, {stmt.min} + {stmt.extent})"
            )
            self._walk_stmt(stmt.body)
            self._isa_lines.append(f"; end for {stmt.loop_var.name_hint}")
        elif isinstance(stmt, tir.Evaluate):
            self._walk_evaluate(stmt)
        elif isinstance(stmt, tir.AttrStmt):
            self._walk_stmt(stmt.body)
        else:
            # Unknown stmt -- emit a comment so we can spot it during dev.
            self._isa_lines.append(f"; <unhandled stmt: {type(stmt).__name__}>")

    def _walk_evaluate(self, ev: tir.Evaluate) -> None:
        val = ev.value
        if not isinstance(val, tir.Call):
            return
        name = self._call_extern_name(val)
        if name is None or not name.startswith("plena."):
            return
        spec = _intrin.lookup(name)
        # call_extern args are: [StringImm(name), op1, op2, ...]
        raw_args = list(val.args[1:])
        resolved, scopes = self._resolve_args(raw_args)
        self._verify_scopes(spec, name, scopes)
        self._isa_lines.append(spec.emit(resolved))

    @staticmethod
    def _call_extern_name(call: tir.Call) -> Optional[str]:
        op = call.op
        # tvm.ir.Op for builtins like "tir.call_extern"
        op_name = getattr(op, "name", None)
        if op_name != "tir.call_extern":
            return None
        if not call.args:
            return None
        head = call.args[0]
        if isinstance(head, tir.StringImm):
            return head.value
        return None

    def _resolve_args(self, args) -> tuple[list[str], list[Optional[str]]]:
        resolved: list[str] = []
        scopes: list[Optional[str]] = []
        for a in args:
            if isinstance(a, tir.Var) and a in self._buffers:
                info = self._buffers[a]
                resolved.append(info.name)
                scopes.append(info.scope)
            elif isinstance(a, tir.BufferLoad) and a.buffer.data in self._buffers:
                info = self._buffers[a.buffer.data]
                if info.scope == _scope.FPRAM:
                    idx = ", ".join(str(self._normalize_scalar_expr(i)) for i in a.indices)
                    resolved.append(f"{info.name}[{idx}]")
                    scopes.append(None)
                else:
                    resolved.append(str(a))
                    scopes.append(None)
            elif isinstance(a, (tir.IntImm, tir.FloatImm)):
                resolved.append(str(a.value))
                scopes.append(None)
            elif isinstance(a, tir.StringImm):
                resolved.append(repr(a.value))
                scopes.append(None)
            else:
                # Could be a buffer .data we missed, or a complex expr.
                # Fall back to a textual rendering and no scope.
                resolved.append(str(a))
                scopes.append(None)
        return resolved, scopes

    @staticmethod
    def _normalize_scalar_expr(a):
        if isinstance(a, tir.IntImm):
            return int(a.value)
        if isinstance(a, tir.FloatImm):
            return float(a.value)
        if isinstance(a, tir.StringImm):
            return str(a.value)
        if isinstance(a, tir.PrimExpr):
            return a
        return str(a)

    def _verify_scopes(
        self, spec: _intrin.IntrinsicSpec, name: str, scopes: list[Optional[str]]
    ) -> None:
        expected = list(spec.operand_scopes)
        if len(scopes) != len(expected):
            raise CodegenError(
                f"{name}: expected {len(expected)} operands, got {len(scopes)}"
            )
        for i, (want, got) in enumerate(zip(expected, scopes)):
            if want is None:
                continue
            if got is None:
                raise CodegenError(
                    f"{name}: operand {i} must be a buffer in scope {want!r}, "
                    f"got non-buffer value"
                )
            if got != want:
                raise CodegenError(
                    f"{name}: operand {i} must be in scope {want!r}, "
                    f"but found {got!r}"
                )

    # ------------------------------------------------------------------
    # header / buffer directives
    # ------------------------------------------------------------------
    def _emit_header(self) -> None:
        self._isa_lines.append(f"; ============================================")
        self._isa_lines.append(f"; PLENA pseudo-ISA  --  kernel: {self.name}")
        self._isa_lines.append(f"; generated by tilelang_tvm_compiler (skeleton)")
        self._isa_lines.append(f"; ============================================")

    def _emit_buffer_directives(self) -> None:
        if not self._buffers:
            return
        self._isa_lines.append("")
        self._isa_lines.append("; ---- buffers ----")
        # Stable order: params first (by appearance), then allocs (by name).
        seen = set()
        order: list[_BufferInfo] = []
        for var in self.func.params:
            buf = self.func.buffer_map.get(var, None)
            if buf is not None and buf.name in self._buffers_by_name:
                info = self._buffers_by_name[buf.name]
                if info.name not in seen:
                    order.append(info)
                    seen.add(info.name)
        for name, info in sorted(self._buffers_by_name.items()):
            if name not in seen:
                order.append(info)
                seen.add(name)
        for info in order:
            shape_str = "x".join(str(s) for s in info.shape)
            scope_token = {
                _scope.HBM: "ALLOC_HBM ",
                _scope.VRAM: "ALLOC_VRAM",
                _scope.MRAM: "ALLOC_MRAM",
                _scope.FPRAM: "ALLOC_FPRAM",
            }[info.scope]
            self._isa_lines.append(
                f"{scope_token}  {info.name}  shape={shape_str}  dtype={info.dtype}"
            )


def compile_module(mod: tvm.IRModule) -> Dict[str, str]:
    """Compile every PrimFunc in `mod` to PLENA pseudo-ISA.

    Returns a {global_symbol -> isa_text} mapping.
    """
    out: Dict[str, str] = {}
    for gv, func in mod.functions.items():
        if not isinstance(func, tir.PrimFunc):
            continue
        cg = PlenaCodegen(func, name=gv.name_hint)
        out[gv.name_hint] = cg.run()
    return out
