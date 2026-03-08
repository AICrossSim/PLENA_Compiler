"""
Compiler-side implementation of Qwen3VLVisionMLP using PLENAProgram and DeveloperCompiler.

Reference module from transformers:

    class Qwen3VLVisionMLP(nn.Module):
        def __init__(self, config):
            self.linear_fc1 = nn.Linear(hidden_size, intermediate_size, bias=True)
            self.linear_fc2 = nn.Linear(intermediate_size, hidden_size, bias=True)
            self.act_fn = ACT2FN[config.hidden_act]

        def forward(self, hidden_state):
            return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))

This file does not implement a PyTorch runtime module. It builds PLENA ISA for the same
high-level computation under the current asm_lib constraints.

Important constraints in the current PLENA stack:
- All matrix dimensions must be multiples of MLEN.
- Bias is compiled as a broadcast matrix in HBM because asm_lib currently lacks a native
  column-vector bias add primitive.
- GELU(Tanh) is emitted from existing tile-row ops and requires a few scalar constants to
  be preloaded in FPRAM.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = CURRENT_DIR.parent.parent.parent
sys.path.insert(0, str(CURRENT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from plena_program import InputVar, PLENAProgram, VRAMMatrixVar


@dataclass(frozen=True)
class Qwen3VLVisionMLPSpec:
    """Minimal config subset required by Qwen3VLVisionMLP."""

    hidden_size: int
    intermediate_size: int
    hidden_act: str = "gelu_pytorch_tanh"


@dataclass(frozen=True)
class GELUTanhFPRAMConstants:
    """
    FPRAM addresses for scalar constants used by gelu_pytorch_tanh.

    Expected mathematical values:
    - one: 1.0
    - half: 0.5
    - two: 2.0
    - neg_one: -1.0
    - gelu_scale: sqrt(2 / pi)
    - gelu_cubic: 0.044715

    The caller is responsible for preloading these scalar values into FPRAM.
    """

    one: int
    half: int
    two: int
    neg_one: int
    gelu_scale: int
    gelu_cubic: int


@dataclass
class Qwen3VLVisionMLPArtifacts:
    """Returned compilation bundle."""

    program: PLENAProgram
    hidden_state: InputVar
    linear_fc1_weight: InputVar
    linear_fc2_weight: InputVar
    output: VRAMMatrixVar
    linear_fc1_bias: Optional[InputVar] = None
    linear_fc2_bias: Optional[InputVar] = None

    @property
    def asm(self) -> str:
        return self.program.compile()


class Qwen3VLVisionMLPCompiler:
    """
    Build PLENA ISA for Qwen3VLVisionMLP.

    Implementation strategy:
    - Use PLENAProgram for object registration, allocation, and final code retrieval.
    - Use DeveloperCompiler directly inside linear/activation kernels to reuse MRAM-loaded
      weight columns and tile-wise elementwise ops.

    Bias convention:
    - The original HF module uses 1D bias vectors.
    - This compiler expects each bias to be stored in HBM as either:
      1. a full broadcast matrix of shape (seq_len, out_features), or
      2. a reusable row-block broadcast matrix of shape (mlen, out_features).
    """

    def __init__(
        self,
        spec: Qwen3VLVisionMLPSpec,
        mlen: int = 64,
        blen: int = 4,
        real_data_ratio: float = 1.125,
        gelu_constants: Optional[GELUTanhFPRAMConstants] = None,
    ):
        self.spec = spec
        self.mlen = mlen
        self.blen = blen
        self.real_data_ratio = real_data_ratio
        self.gelu_constants = gelu_constants
        self._tile_rows = list(range(self.mlen))

    def build(
        self,
        seq_len: int,
        use_fc1_bias: bool = True,
        use_fc2_bias: bool = True,
    ) -> Qwen3VLVisionMLPArtifacts:
        """Create a PLENAProgram and compile one Qwen3VLVisionMLP instance."""
        self._validate_dim("seq_len", seq_len)
        self._validate_dim("hidden_size", self.spec.hidden_size)
        self._validate_dim("intermediate_size", self.spec.intermediate_size)

        prog = PLENAProgram(
            mlen=self.mlen,
            blen=self.blen,
            real_data_ratio=self.real_data_ratio,
        )

        hidden_state = prog.input("hidden_state", shape=(seq_len, self.spec.hidden_size))
        linear_fc1_weight = prog.input(
            "linear_fc1_weight",
            shape=(self.spec.hidden_size, self.spec.intermediate_size),
        )
        linear_fc2_weight = prog.input(
            "linear_fc2_weight",
            shape=(self.spec.intermediate_size, self.spec.hidden_size),
        )

        linear_fc1_bias = None
        if use_fc1_bias:
            linear_fc1_bias = prog.input(
                "linear_fc1_bias_block",
                shape=(self.mlen, self.spec.intermediate_size),
            )

        linear_fc2_bias = None
        if use_fc2_bias:
            linear_fc2_bias = prog.input(
                "linear_fc2_bias_block",
                shape=(self.mlen, self.spec.hidden_size),
            )

        hidden_vram = prog.load_batch(hidden_state, name="hidden_state")

        fc1_out = self._compile_linear(
            prog=prog,
            source=hidden_vram,
            weight=linear_fc1_weight,
            out_features=self.spec.intermediate_size,
            output_name="linear_fc1_out",
            bias=linear_fc1_bias,
        )
        prog.free_tensor(hidden_vram)

        activated = self._apply_activation(
            prog=prog,
            source=fc1_out,
            output_name="vision_mlp_act",
        )
        if activated.name != fc1_out.name:
            prog.free_tensor(fc1_out)

        output = self._compile_linear(
            prog=prog,
            source=activated,
            weight=linear_fc2_weight,
            out_features=self.spec.hidden_size,
            output_name="vision_mlp_out",
            bias=linear_fc2_bias,
        )
        prog.free_tensor(activated)

        prog.result(output)

        return Qwen3VLVisionMLPArtifacts(
            program=prog,
            hidden_state=hidden_state,
            linear_fc1_weight=linear_fc1_weight,
            linear_fc2_weight=linear_fc2_weight,
            linear_fc1_bias=linear_fc1_bias,
            linear_fc2_bias=linear_fc2_bias,
            output=output,
        )

    def _compile_linear(
        self,
        prog: PLENAProgram,
        source: VRAMMatrixVar,
        weight: InputVar,
        out_features: int,
        output_name: str,
        bias: Optional[InputVar] = None,
    ) -> VRAMMatrixVar:
        """Compile a dense layer with optional broadcast bias."""
        rows, in_features = source.shape
        if weight.shape != (in_features, out_features):
            raise ValueError(
                f"Linear weight shape mismatch for {output_name}: "
                f"expected {(in_features, out_features)}, got {weight.shape}"
            )

        output = prog.alloc(output_name, rows, out_features)
        compiler = prog.compiler

        compiler.ensure_vram_matrix_layout(source.name, source.shape)
        compiler.ensure_vram_matrix_layout(output.name, output.shape)
        compiler.ensure_hbm_sub_matrix(
            name=weight.name,
            hbm_addr=weight.hbm_addr,
            shape=weight.shape,
            real_data_ratio=self.real_data_ratio,
        )

        num_row_blocks = rows // self.mlen
        num_col_blocks = out_features // self.mlen

        compiler.generated_code += f"; === Linear {output_name}: {source.name} @ {weight.name} ===\n"
        for col_idx in range(num_col_blocks):
            compiler.reset_mram()
            compiler.load_sub_matrix_col(name=weight.name, col_idx=col_idx)
            for row_idx in range(num_row_blocks):
                compiler.vram_sub_projection_to(
                    vram_mat_name=source.name,
                    vram_row_idx=row_idx,
                    mram_mat_name=weight.name,
                    mram_col_idx=col_idx,
                    target_matrix=output.name,
                    target_row_idx=row_idx,
                    target_col_idx=col_idx,
                )

        if bias is not None:
            self._apply_bias(prog, output, bias)

        return output

    def _apply_bias(
        self,
        prog: PLENAProgram,
        target: VRAMMatrixVar,
        bias: InputVar,
    ):
        """Apply bias stored as a full matrix or as a reusable row-block matrix."""
        rows, cols = target.shape
        if bias.shape[1] != cols:
            raise ValueError(
                f"Bias width mismatch for {target.name}: expected {cols}, got {bias.shape[1]}"
            )

        bias_matrix = prog.load_batch(bias, name=f"{target.display_name}_bias")

        if bias.shape == target.shape:
            prog.vram_add(target, bias_matrix)
        elif bias.shape == (self.mlen, cols):
            num_row_blocks = rows // self.mlen
            for row_block in range(num_row_blocks):
                prog.vram_add(
                    target,
                    bias_matrix,
                    dst_row_offset=row_block * self.mlen,
                    src_row_offset=0,
                    num_rows=self.mlen,
                )
        else:
            raise ValueError(
                f"Unsupported bias shape for {target.name}: {bias.shape}. "
                f"Expected {(rows, cols)} or {(self.mlen, cols)}."
            )

        prog.free_tensor(bias_matrix)

    def _apply_activation(
        self,
        prog: PLENAProgram,
        source: VRAMMatrixVar,
        output_name: str,
    ) -> VRAMMatrixVar:
        """Compile the Qwen3VLVisionMLP activation."""
        act = self.spec.hidden_act
        if act in ("identity", None):
            return source
        if act not in ("gelu", "gelu_pytorch_tanh"):
            raise NotImplementedError(
                f"Activation '{act}' is not supported yet by this PLENA compiler."
            )
        if self.gelu_constants is None:
            raise ValueError(
                "gelu_pytorch_tanh requires GELUTanhFPRAMConstants because asm_lib "
                "cannot materialize floating-point immediates by itself."
            )

        return self._apply_gelu_pytorch_tanh(prog, source, output_name)

    def _apply_gelu_pytorch_tanh(
        self,
        prog: PLENAProgram,
        source: VRAMMatrixVar,
        output_name: str,
    ) -> VRAMMatrixVar:
        """
        GELU(Tanh) approximation:

            gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        """
        square = self._copy_matrix(prog, source, f"{output_name}_square")
        self._matrix_mul_inplace(prog, square, source)

        output = self._copy_matrix(prog, square, output_name)
        self._matrix_mul_inplace(prog, output, source)
        prog.free_tensor(square)

        self._matrix_mul_scalar_inplace(prog, output, self.gelu_constants.gelu_cubic)
        self._matrix_add_inplace(prog, output, source)
        self._matrix_mul_scalar_inplace(prog, output, self.gelu_constants.gelu_scale)
        self._tanh_inplace(prog, output)
        self._matrix_add_scalar_inplace(prog, output, self.gelu_constants.one)
        self._matrix_mul_scalar_inplace(prog, output, self.gelu_constants.half)
        self._matrix_mul_inplace(prog, output, source)
        return output

    def _tanh_inplace(self, prog: PLENAProgram, matrix: VRAMMatrixVar):
        """tanh(x) = 2 / (1 + exp(-2x)) - 1"""
        self._matrix_mul_scalar_inplace(prog, matrix, self.gelu_constants.two)
        self._matrix_mul_scalar_inplace(prog, matrix, self.gelu_constants.neg_one)
        self._matrix_exp_inplace(prog, matrix)
        self._matrix_add_scalar_inplace(prog, matrix, self.gelu_constants.one)
        self._matrix_reci_inplace(prog, matrix)
        self._matrix_mul_scalar_inplace(prog, matrix, self.gelu_constants.two)
        self._matrix_add_scalar_inplace(prog, matrix, self.gelu_constants.neg_one)

    def _copy_matrix(
        self,
        prog: PLENAProgram,
        source: VRAMMatrixVar,
        name: str,
    ) -> VRAMMatrixVar:
        """Copy a VRAM matrix via zero-fill followed by matrix add."""
        copied = prog.alloc(name, source.shape[0], source.shape[1])
        self._matrix_zero_inplace(prog, copied)
        prog.compiler.vram_matrix_add(
            dst_matrix=copied.name,
            src_matrix=source.name,
        )
        return copied

    def _matrix_zero_inplace(self, prog: PLENAProgram, matrix: VRAMMatrixVar):
        compiler = prog.compiler
        num_row_blocks = matrix.shape[0] // self.mlen
        num_col_blocks = matrix.shape[1] // self.mlen
        for row_block in range(num_row_blocks):
            for col_block in range(num_col_blocks):
                compiler.vram_fill_zero(
                    matrix_name=matrix.name,
                    rows=self._tile_rows,
                    tile_row_idx=row_block,
                    tile_col_idx=col_block,
                )

    def _matrix_add_inplace(
        self,
        prog: PLENAProgram,
        dst: VRAMMatrixVar,
        src: VRAMMatrixVar,
    ):
        prog.compiler.vram_matrix_add(
            dst_matrix=dst.name,
            src_matrix=src.name,
        )

    def _matrix_mul_inplace(
        self,
        prog: PLENAProgram,
        dst: VRAMMatrixVar,
        src: VRAMMatrixVar,
    ):
        if dst.shape != src.shape:
            raise ValueError(f"Elementwise mul shape mismatch: {dst.shape} vs {src.shape}")

        compiler = prog.compiler
        num_row_blocks = dst.shape[0] // self.mlen
        num_col_blocks = dst.shape[1] // self.mlen
        for row_block in range(num_row_blocks):
            for col_block in range(num_col_blocks):
                compiler.tile_row_mul(
                    dst_matrix=dst.name,
                    src_matrix=src.name,
                    rows=self._tile_rows,
                    dst_tile_row_idx=row_block,
                    dst_tile_col_idx=col_block,
                    src_tile_row_idx=row_block,
                    src_tile_col_idx=col_block,
                )

    def _matrix_mul_scalar_inplace(
        self,
        prog: PLENAProgram,
        matrix: VRAMMatrixVar,
        scalar_fpram_addr: int,
    ):
        compiler = prog.compiler
        num_row_blocks = matrix.shape[0] // self.mlen
        num_col_blocks = matrix.shape[1] // self.mlen
        for row_block in range(num_row_blocks):
            for col_block in range(num_col_blocks):
                compiler.tile_row_mul_fp_broadcast(
                    matrix_name=matrix.name,
                    fpram_scalar_addr=scalar_fpram_addr,
                    rows=self._tile_rows,
                    tile_row_idx=row_block,
                    tile_col_idx=col_block,
                )

    def _matrix_add_scalar_inplace(
        self,
        prog: PLENAProgram,
        matrix: VRAMMatrixVar,
        scalar_fpram_addr: int,
    ):
        compiler = prog.compiler
        row_map = [(row_idx, scalar_fpram_addr) for row_idx in self._tile_rows]
        num_row_blocks = matrix.shape[0] // self.mlen
        num_col_blocks = matrix.shape[1] // self.mlen
        for row_block in range(num_row_blocks):
            for col_block in range(num_col_blocks):
                compiler.tile_row_add_fp(
                    matrix_name=matrix.name,
                    row_map=row_map,
                    tile_row_idx=row_block,
                    tile_col_idx=col_block,
                )

    def _matrix_exp_inplace(self, prog: PLENAProgram, matrix: VRAMMatrixVar):
        compiler = prog.compiler
        num_row_blocks = matrix.shape[0] // self.mlen
        num_col_blocks = matrix.shape[1] // self.mlen
        for row_block in range(num_row_blocks):
            for col_block in range(num_col_blocks):
                compiler.tile_row_exp(
                    matrix_name=matrix.name,
                    rows=self._tile_rows,
                    tile_row_idx=row_block,
                    tile_col_idx=col_block,
                )

    def _matrix_reci_inplace(self, prog: PLENAProgram, matrix: VRAMMatrixVar):
        compiler = prog.compiler
        num_row_blocks = matrix.shape[0] // self.mlen
        num_col_blocks = matrix.shape[1] // self.mlen
        for row_block in range(num_row_blocks):
            for col_block in range(num_col_blocks):
                compiler.tile_row_reci(
                    matrix_name=matrix.name,
                    rows=self._tile_rows,
                    tile_row_idx=row_block,
                    tile_col_idx=col_block,
                )

    def _validate_dim(self, name: str, value: int):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
        if value % self.mlen != 0:
            raise ValueError(
                f"{name}={value} must be a multiple of mlen={self.mlen} for the current asm_lib."
            )


def build_qwen3_vl_vision_mlp_program(
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
    hidden_act: str = "gelu_pytorch_tanh",
    mlen: int = 64,
    blen: int = 4,
    real_data_ratio: float = 1.125,
    gelu_constants: Optional[GELUTanhFPRAMConstants] = None,
    use_fc1_bias: bool = True,
    use_fc2_bias: bool = True,
) -> Qwen3VLVisionMLPArtifacts:
    """Convenience entry point."""
    compiler = Qwen3VLVisionMLPCompiler(
        spec=Qwen3VLVisionMLPSpec(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
        ),
        mlen=mlen,
        blen=blen,
        real_data_ratio=real_data_ratio,
        gelu_constants=gelu_constants,
    )
    return compiler.build(
        seq_len=seq_len,
        use_fc1_bias=use_fc1_bias,
        use_fc2_bias=use_fc2_bias,
    )
