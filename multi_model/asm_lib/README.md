# PLENA ASM Lib Usage Guide

This document explains the responsibilities, use cases, and recommended usage patterns of `plena_program.py`, `developer_compiler.py`, and `sub_matrix_manager.py`.

## 1. Layered Roles of the Three Files

### `plena_program.py`
- This is the upper-level API for people who are "writing programs".
- It provides a Python-style DSL that wraps HBM inputs, VRAM tensors, FPRAM scalars, and common operators as objects and methods.
- In most cases, you should start from `PLENAProgram` rather than writing `DeveloperCompiler` code directly.
- Typical responsibilities:
  - declare inputs: `input()`
  - load from HBM to VRAM: `load_batch()`
  - allocate intermediate results: `alloc()`, `fp_var()`
  - organize operators: `norm()`, `vram_sub_projection_to()`, `vram_add()`, `tile_row_*()`
  - manage scopes and reusable functions: `@prog.function`
  - export ISA: `compile()`

### `developer_compiler.py`
- This is the lower-level compiler backend for people who are "writing compilers" or "implementing low-level kernels".
- It directly manages:
  - `SubMatrixManager` / symbol table
  - VRAM / HBM / FPRAM addresses
  - GP / Address / FP register allocation
  - ISA template assembly and code accumulation
- Typical responsibilities:
  - lower higher-level operations into concrete ISA
  - ensure that addresses and registers used by load/store/projection/softmax/FPRAM operations are valid
  - provide lower-level address-based helpers such as `fpvar_*_asm()` and `tile_row_*_asm()`

### `sub_matrix_manager.py`
- This is the layout and address-calculation core beneath `PLENAProgram` and `DeveloperCompiler`.
- It does not manage program flow; it defines how objects are organized and addressed in HBM / VRAM / MRAM / FPRAM.
- Typical responsibilities:
  - register HBM / VRAM / FPRAM objects
  - split large matrices into `64 x 64` sub-blocks
  - precompute HBM offsets, VRAM addresses, and MRAM load positions for each block
  - manage `VRAMAllocator`, `MRAMAllocator`, and `FPRAMAllocator`
  - generate low-level ISA for block loads and block multiplication, such as:
    - `load_sub_matrix_asm()`
    - `load_row_sub_matrices_asm()`
    - `load_col_sub_matrices_asm()`
    - `vram_sub_projection_asm()`
    - `vram_sub_projection_T_asm()`
    - `vram_block_add_asm()`

### Recommended Mental Model
- `PLENAProgram` = programming interface / scheduling layer
- `DeveloperCompiler` = ISA generator / resource-management layer
- `SubMatrixManager` = layout, tiling, and address backend

## 2. How You Should Use These Three Files

### Preferred Usage: Only Use `PLENAProgram`
If your goal is to describe an operator flow and generate ISA, prefer `PLENAProgram`.

Recommended workflow:
1. Create a program object: `prog = PLENAProgram(mlen=64, blen=4)`
2. Declare HBM inputs: `prog.input(...)`
3. Load inputs that participate in computation into VRAM: `prog.load_batch(...)`
4. Allocate VRAM/FPRAM for intermediate results: `prog.alloc(...)`, `prog.fp_var(...)`
5. Organize computation through the high-level operator APIs
6. Call `prog.compile()` to get the final ISA string

Benefits:
- fewer address-level details
- no need to manually manage register release
- safer namespaces and temporary tensor lifetimes

### Advanced Usage: Drop Down Through `prog.compiler`
If the high-level API is missing an operation, or you need to validate the behavior of a specific ISA template, you can enter `DeveloperCompiler` via `prog.compiler`.

Suitable when:
- you want to add a new high-level API, but only a low-level kernel exists so far
- you need to debug the symbol table, VRAM addresses, or FPRAM addresses
- you want to assemble a small `*_asm` helper sequence directly

Recommended pattern:
- still use `PLENAProgram` to manage inputs and object lifetimes
- only move the missing fragment down into `prog.compiler`
- do not bypass `PLENAProgram` for the whole program, otherwise you must manually handle object registration, naming, and resource release

### Lower-Level Usage: Directly Use `SubMatrixManager`
Direct interaction with `SubMatrixManager` is recommended only when:
- you need to debug whether block addresses are correct
- you want to verify the layout difference between HBM row-major storage and VRAM column-block-major layout
- you want to add a new block-level ISA helper
- you need to inspect allocator reuse, fragmentation recovery, or alignment behavior

It is usually not recommended as the main entry point because it does not represent high-level program semantics; it only manages memory objects and block layout.

## 3. Core Interfaces of the Three Files

### Common `PLENAProgram` Entry Points
- `input(name, shape, hbm_addr=None)`
  - declare an input tensor in HBM
- `load_batch(input_var, name=None)`
  - generate load ISA from HBM to VRAM
- `alloc(name, rows, cols)`
  - allocate an intermediate matrix in VRAM
- `store(tensor_var, name=None, hbm_addr=None)`
  - write a VRAM result back to HBM
- `fp_var(name, size=1)`
  - declare a scalar or scalar array in FPRAM
- `norm()`, `rms_norm()`, `layer_norm()`
  - normalize a VRAM tensor in place
- `tile_row_max/sum/exp/reci/sub_fp/mul_fp/...`
  - tile-row reductions or per-row scalar operations
- `vram_sub_projection_to()`
  - `A[row_block] @ W[:, col_block]`
- `vram_sub_projection_T_to()`
  - `A[row_block] @ W[row_block]^T`
- `vram_add()`, `vram_block_add_to()`
  - matrix or block accumulation
- `init_online_softmax()`, `online_softmax_block()`, `compute_pv()`, `scale_o_row()`, `final_scale_o()`
  - dedicated flow for flash attention
- `compile()`
  - return the accumulated ISA

### Common `DeveloperCompiler` Entry Points
- `add_hbm_object()`
- `load_batch()`
- `store_to_hbm()`
- `allocate_vram_matrix()`
- `allocate_fpram()`
- `tile_row_*()`
- `fpram_*()`
- `vram_sub_projection_to()`
- `vram_sub_projection_T_to()`
- `init_online_softmax()`
- `online_softmax_block()`
- `compute_pv()`
- `get_code()`

These interfaces are lower-level and usually take internal names, addresses, or block indices as parameters.

### Common `SubMatrixManager` Entry Points
- `add_hbm_object()`
- `add_vram_object()`
- `add_fpram_object()`
- `free_hbm_object()`
- `free_vram_object()`
- `free_fpram_object()`
- `register_matrix()`
- `register_vram_matrix()`
- `get_sub_block()`
- `get_row_blocks()`
- `get_col_blocks()`
- `get_vram_sub_block()`
- `compute_hbm_offset()`
- `compute_absolute_hbm_addr()`
- `load_sub_matrix_asm()`
- `load_row_sub_matrices_asm()`
- `load_col_sub_matrices_asm()`
- `vram_sub_projection_asm()`
- `vram_sub_projection_T_asm()`
- `vram_block_add_asm()`
- `generate_address_table()`
- `print_table()`
- `print_layout()`

The value of these interfaces is not in expressing algorithms, but in expressing layout and addressing.

## 4. Practical Development Boundaries

### When You Are Only Orchestrating Models or Operators
Use `PLENAProgram`.

### When You Are Adding a New Operator
Recommended order:
1. First check whether `SubMatrixManager` already provides a suitable block layout and block-level ISA helper
2. If not, add the address/block-level capability in `SubMatrixManager` first
3. Wrap that capability into a more complete compilation step in `DeveloperCompiler`
4. Add a user-friendly API layer in `PLENAProgram`
5. Validate it from test files through `PLENAProgram`

This is the most natural extension path for the current directory structure.

## 5. Key Capabilities of `sub_matrix_manager.py`

### 1. Unified Object Model
`SubMatrixManager.__getitem__()` merges HBM / VRAM / FPRAM information into a unified `MemoryObjectInfo` view.

This means:
- the same name can simultaneously have both HBM and VRAM layouts
- `DeveloperCompiler` does not need to maintain multiple separate tables
- `PLENAProgram.symbol_table` is essentially viewing this structure

### 2. Three Allocator Types
- `VRAMAllocator`
  - manages contiguous VRAM space aligned to `MLEN=64`
- `MRAMAllocator`
  - manages MRAM sub-block space aligned to `MLEN * MLEN = 4096`
- `FPRAMAllocator`
  - manages FPRAM scalar space and supports `save_state()` / `restore_state()`

They are all built on the `VirtualMemoryManager` model of best-fit + free-list + coalescing.

### 3. The Layout Difference Between HBM and VRAM Is Explicitly Encoded
- HBM: 2D matrices are stored contiguously in row-major order
- VRAM: organized in a column-block-major format of `[batch, mlen, hidden/mlen]`

This is exactly why:
- HBM sub-block addresses mainly use `hbm_base + hbm_offset`
- VRAM sub-block addresses mainly use `vram_base + col_block * batch * mlen + row_block * mlen * mlen`

### 4. Sub-Block Addressing Capability
`MatrixBlockLayout` and `VRAMMatrixBlockLayout` pre-split a full matrix into multiple `64 x 64` sub-blocks and record:
- which parent matrix it belongs to
- row-block / col-block indices
- HBM offset or VRAM address
- whether it has already been loaded into MRAM

This lets `DeveloperCompiler` generate ISA directly from block metadata instead of calculating addresses manually at runtime.

### 5. Block-Level ISA Generation
`SubMatrixManager` already directly provides low-level block-operation code generation for:
- block loading from HBM to MRAM
- loading an entire row of blocks or an entire column of blocks
- `VRAM row-block @ MRAM col-block`
- `VRAM row-block @ MRAM row-block^T`
- `VRAM block + VRAM block`

So many `DeveloperCompiler` methods are essentially doing:
1. allocate/register layouts
2. verify blocks are loaded
3. call `SubMatrixManager`'s `*_asm()`
4. release registers and accumulate code

## 6. Minimal Example

```python
from plena_program import PLENAProgram

prog = PLENAProgram(mlen=64, blen=4)

# 1. HBM inputs
x = prog.input("X", shape=(128, 128))
w = prog.input("W", shape=(128, 128))

# 2. Load the required inputs into VRAM
x_vram = prog.load_batch(x, name="Xv")

# 3. Allocate the output matrix
y = prog.alloc("Y", 128, 128)

# 4. Run one sub-block projection: Y[0][0] = Xv[0][:] @ W[:][0]
prog.vram_sub_projection_to(
    vram_matrix=x_vram,
    vram_row_idx=0,
    mram_input=w,
    mram_col_idx=0,
    target=y,
    target_row_idx=0,
    target_col_idx=0,
)

# 5. Export ISA
asm = prog.compile()
print(asm)
```
