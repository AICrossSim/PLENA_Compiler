"""Mamba-2 reference semantics -- the golden the PLENA lowering is checked against.

Three implementations, deliberately kept side by side:

``mamba2_recurrent_reference``
    The textbook per-timestep recurrence. Slow, obviously correct, no chunking.
    This is the definition; everything else is checked against it.

``mamba2_chunked_reference``
    The SSD chunked form the PLENA prefill lowering actually implements, written
    in the same order with the same intermediates. When the emulator disagrees
    with the golden, comparing against *this* rather than against the recurrence
    localises the failure to a stage instead of to "the layer".

``mamba2_reference``
    The full mixer: in_proj, conv1d, dt activation, scan, gated norm, out_proj.

The two scan implementations must agree to floating-point tolerance;
``test_mamba2_reference.py`` asserts that, which is what makes the chunked form
usable as an intermediate golden at all.

A note on what "golden" means here
----------------------------------
These run in float32 and are the *algorithmic* reference. They are NOT a model of
PLENA's arithmetic. In particular the emulator implements ``V_EXP_V`` as
libtorch's exact ``exp``, while the RTL model in
``PLENA_Tools/plena_quant/quant_operations/exp.py`` is a fixed-point range
reduction plus a 3-term Taylor with roughly 0.5-1.5% *systematic* relative error.
Attention's softmax normalises such a bias away; Mamba applies ``exp`` on the
critical path of a multiplicative recurrence, where it compounds. So a Mamba
accuracy result measured on the transactional emulator bounds the *lowering*, not
the silicon. Closing that gap means plumbing the hardware exp model into the
comparison, and is deliberately left as a separate, visible piece of work rather
than papered over with a loose tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Mamba2Params:
    """Weights and biases of one Mamba-2 mixer, in HuggingFace orientation."""

    in_proj_weight: torch.Tensor  # [in_proj_out, hidden]
    conv_weight: torch.Tensor  # [conv_dim, conv_kernel]
    conv_bias: torch.Tensor | None  # [conv_dim]
    dt_bias: torch.Tensor  # [num_heads]
    A_log: torch.Tensor  # [num_heads]
    D: torch.Tensor  # [num_heads]
    norm_weight: torch.Tensor | None  # [d_inner]
    out_proj_weight: torch.Tensor  # [hidden, d_inner]

    num_heads: int
    head_dim: int
    state_size: int
    n_groups: int
    conv_kernel: int
    chunk_size: int
    time_step_min: float = 0.0
    time_step_max: float = float("inf")
    norm_eps: float = 1e-5

    @property
    def d_inner(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def group_state(self) -> int:
        return self.n_groups * self.state_size

    @property
    def conv_dim(self) -> int:
        return self.d_inner + 2 * self.group_state


@dataclass
class Mamba2Result:
    """Layer output plus every intermediate a stage-level test needs."""

    y: torch.Tensor  # [batch, seq, hidden]
    x: torch.Tensor  # [batch, seq, num_heads, head_dim]  (post conv+silu)
    B: torch.Tensor  # [batch, seq, n_groups, state_size]
    C: torch.Tensor  # [batch, seq, n_groups, state_size]
    dt: torch.Tensor  # [batch, seq, num_heads]          (post softplus+clamp)
    z: torch.Tensor  # [batch, seq, d_inner]
    scan_out: torch.Tensor  # [batch, seq, num_heads, head_dim]
    final_state: torch.Tensor  # [batch, num_heads, state_size, head_dim]


def random_mamba2_params(
    *,
    hidden_size: int,
    num_heads: int,
    head_dim: int,
    state_size: int,
    n_groups: int = 1,
    conv_kernel: int = 4,
    chunk_size: int = 64,
    use_conv_bias: bool = True,
    use_norm_weight: bool = True,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
) -> Mamba2Params:
    """Small, well-conditioned random parameters for tests.

    ``A_log`` is drawn so that ``A = -exp(A_log)`` lands in roughly ``[-4, -1]``:
    the HF initialisation range, and deliberately not near zero, because
    ``A -> 0`` makes the decay ~1 and hides exactly the error accumulation this
    lowering has to get right.
    """
    g = torch.Generator().manual_seed(seed)
    d_inner = num_heads * head_dim
    group_state = n_groups * state_size
    in_proj_out = 2 * d_inner + 2 * group_state + num_heads
    conv_dim = d_inner + 2 * group_state

    def rnd(*shape, scale=0.1):
        return (torch.randn(*shape, generator=g, dtype=dtype) * scale).contiguous()

    return Mamba2Params(
        in_proj_weight=rnd(in_proj_out, hidden_size),
        conv_weight=rnd(conv_dim, conv_kernel, scale=0.3),
        conv_bias=rnd(conv_dim, scale=0.05) if use_conv_bias else None,
        dt_bias=rnd(num_heads, scale=0.5),
        A_log=torch.log(torch.rand(num_heads, generator=g, dtype=dtype) * 3.0 + 1.0),
        D=rnd(num_heads, scale=0.5),
        norm_weight=torch.ones(d_inner, dtype=dtype) + rnd(d_inner, scale=0.05)
        if use_norm_weight
        else None,
        out_proj_weight=rnd(hidden_size, d_inner),
        num_heads=num_heads,
        head_dim=head_dim,
        state_size=state_size,
        n_groups=n_groups,
        conv_kernel=conv_kernel,
        chunk_size=chunk_size,
    )


def _silu(t: torch.Tensor) -> torch.Tensor:
    return t * torch.sigmoid(t)


def _causal_depthwise_conv1d(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None
) -> torch.Tensor:
    """`x` is [batch, seq, channels]; `weight` is [channels, kernel].

    Written as an explicit tap sum rather than ``F.conv1d`` so it mirrors the
    lowering in ``program_mamba_common.mamba_conv1d_v0`` line for line -- the
    causal left pad shows up as skipped taps in both.
    """
    batch, seq, channels = x.shape
    kernel = weight.shape[1]
    out = torch.zeros_like(x)
    for s in range(seq):
        for j in range(kernel):
            src = s - (kernel - 1) + j
            if src < 0:
                continue
            out[:, s, :] += x[:, src, :] * weight[:, j]
    if bias is not None:
        out += bias
    return out


def mamba2_recurrent_reference(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The textbook selective-scan recurrence. This is the definition.

    Shapes: `x` [batch, seq, heads, head_dim]; `dt` [batch, seq, heads];
    `A` [heads]; `B`/`C` [batch, seq, groups, state]; `D` [heads].
    Returns (y [batch, seq, heads, head_dim], final_state
    [batch, heads, state, head_dim]).
    """
    batch, seq, heads, head_dim = x.shape
    _, _, groups, state = B.shape
    heads_per_group = heads // groups

    h = (
        torch.zeros(batch, heads, state, head_dim, dtype=x.dtype)
        if initial_state is None
        else initial_state.clone()
    )
    y = torch.zeros_like(x)

    for t in range(seq):
        dA = torch.exp(dt[:, t] * A)  # [batch, heads]
        h = h * dA[:, :, None, None]
        for g in range(groups):
            hs = slice(g * heads_per_group, (g + 1) * heads_per_group)
            dtx = dt[:, t, hs][:, :, None] * x[:, t, hs]  # [batch, heads_g, head_dim]
            # rank-1 update: B[b, n] outer dtx[b, head, p]
            h[:, hs] = h[:, hs] + B[:, t, g][:, None, :, None] * dtx[:, :, None, :]
            y[:, t, hs] = torch.einsum("bhnp,bn->bhp", h[:, hs], C[:, t, g])
        y[:, t] = y[:, t] + D[None, :, None] * x[:, t]
    return y, h


def ssd_chunk_reference(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    chunk_size: int,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The chunked (SSD) form, in the same order the PLENA lowering emits it.

    Each step here corresponds to one emitter in ``program_ssd.py``:
    ``cs`` to ``ssd_chunk_cumsum_v0``, ``L`` to ``ssd_decay_mask_v0``,
    ``Y_intra`` to ``ssd_chunk_head_v0``, the state update to
    ``ssd_state_update_v0``, and ``Y_inter`` to ``ssd_inter_chunk_output_v0``.
    """
    batch, seq, heads, head_dim = x.shape
    _, _, groups, state = B.shape
    heads_per_group = heads // groups

    h = (
        torch.zeros(batch, heads, state, head_dim, dtype=x.dtype)
        if initial_state is None
        else initial_state.clone()
    )
    y = torch.zeros_like(x)

    for start in range(0, seq, chunk_size):
        end = min(start + chunk_size, seq)
        L = end - start
        xc = x[:, start:end]  # [b, L, h, p]
        dtc = dt[:, start:end]  # [b, L, h]
        Bc = B[:, start:end]  # [b, L, g, n]
        Cc = C[:, start:end]

        a = dtc * A  # [b, L, h]
        cs = torch.cumsum(a, dim=1)  # [b, L, h]

        # Intra-chunk: (L o G) @ X, with L[i,j] = exp(cs_i - cs_j), causal.
        # Clamping to <= 0 before exp is what the lowering does and is exact on
        # the causal half (cs is non-increasing because A < 0).
        d_mat = cs[:, :, None, :] - cs[:, None, :, :]  # [b, i, j, h]
        d_mat = torch.clamp(d_mat, max=0.0)
        causal = torch.tril(torch.ones(L, L, dtype=x.dtype))
        decay = torch.exp(d_mat) * causal[None, :, :, None]

        for g in range(groups):
            hs = slice(g * heads_per_group, (g + 1) * heads_per_group)
            G = torch.einsum("bin,bjn->bij", Cc[:, :, g], Bc[:, :, g])  # [b, i, j]
            LG = G[:, :, :, None] * decay[:, :, :, hs]  # [b, i, j, h]
            # dt scales the *source* timestep j
            LG = LG * dtc[:, None, :, hs]
            y[:, start:end, hs] = torch.einsum("bijh,bjhp->bihp", LG, xc[:, :, hs])

            # Inter-chunk: C @ h_prev, row i scaled by exp(cs_i)
            y_inter = torch.einsum("bin,bhnp->bihp", Cc[:, :, g], h[:, hs])
            y[:, start:end, hs] = y[:, start:end, hs] + y_inter * torch.exp(cs[:, :, hs])[:, :, :, None]

            # State: h = h * exp(sum a) + B^T @ (X scaled by exp(cs_end - cs_j) * dt)
            cs_end = cs[:, -1, hs]  # [b, h]
            scale = torch.exp(cs_end[:, None, :] - cs[:, :, hs]) * dtc[:, :, hs]  # [b, L, h]
            xd = xc[:, :, hs] * scale[:, :, :, None]
            h[:, hs] = h[:, hs] * torch.exp(cs_end)[:, :, None, None] + torch.einsum(
                "bjn,bjhp->bhnp", Bc[:, :, g], xd
            )

        y[:, start:end] = y[:, start:end] + D[None, None, :, None] * xc

    return y, h


def mamba2_reference(
    hidden: torch.Tensor,
    params: Mamba2Params,
    *,
    use_chunked: bool = True,
    initial_state: torch.Tensor | None = None,
) -> Mamba2Result:
    """Full Mamba-2 mixer forward pass.

    `hidden` is [batch, seq, hidden_size]. Set ``use_chunked=False`` to run the
    plain recurrence instead of the SSD form.
    """
    p = params
    batch, seq, _ = hidden.shape

    projected = hidden @ p.in_proj_weight.T  # [b, s, in_proj_out]
    d_inner, gs = p.d_inner, p.group_state
    z = projected[..., :d_inner]
    xbc = projected[..., d_inner : d_inner + p.conv_dim]
    dt_raw = projected[..., d_inner + p.conv_dim :]

    xbc = _silu(_causal_depthwise_conv1d(xbc, p.conv_weight, p.conv_bias))
    x_flat = xbc[..., :d_inner]
    B = xbc[..., d_inner : d_inner + gs].reshape(batch, seq, p.n_groups, p.state_size)
    C = xbc[..., d_inner + gs :].reshape(batch, seq, p.n_groups, p.state_size)

    dt = torch.nn.functional.softplus(dt_raw + p.dt_bias)
    dt = torch.clamp(dt, min=p.time_step_min, max=p.time_step_max)

    x = x_flat.reshape(batch, seq, p.num_heads, p.head_dim)
    A = -torch.exp(p.A_log)

    if use_chunked:
        scan_out, final_state = ssd_chunk_reference(
            x, dt, A, B, C, p.D, p.chunk_size, initial_state=initial_state
        )
    else:
        scan_out, final_state = mamba2_recurrent_reference(
            x, dt, A, B, C, p.D, initial_state=initial_state
        )

    y = scan_out.reshape(batch, seq, d_inner)

    # Gated RMSNorm.
    #
    # The gate is applied BEFORE the variance, which is what both upstream
    # implementations do: HuggingFace's `MambaRMSNormGated.forward` multiplies by
    # `silu(gate)` and only then computes `hidden_states.pow(2).mean(-1)`, and
    # mamba_ssm's `Mamba2` constructs `RMSNormGated(..., norm_before_gate=False)`.
    # The two orders are different functions, not different roundings -- RMSNorm is
    # not per-element homogeneous -- and at these shapes they differ by ~56%
    # relative. Getting this backwards would be unrecoverable once real Mamba-2
    # weights are loaded, and would silently retarget every downstream
    # emulator-vs-golden comparison.
    #
    # The reduction is over `d_inner / n_groups`, following mamba_ssm's
    # `group_size=d_ssm // ngroups`. HuggingFace reduces over the whole of
    # `intermediate_size`; the two agree exactly at `n_groups == 1`, which is what
    # mamba2-2.7b and every other published Mamba-2 checkpoint uses.
    y = y * _silu(z)
    group_width = d_inner // p.n_groups
    yg = y.reshape(batch, seq, p.n_groups, group_width)
    rms = torch.rsqrt(yg.pow(2).mean(dim=-1, keepdim=True) + p.norm_eps)
    yg = yg * rms
    y = yg.reshape(batch, seq, d_inner)
    if p.norm_weight is not None:
        y = y * p.norm_weight

    out = y @ p.out_proj_weight.T
    return Mamba2Result(
        y=out, x=x, B=B, C=C, dt=dt, z=z, scan_out=scan_out, final_state=final_state
    )


def mamba2_chunked_reference(*args, **kwargs):
    """Alias kept so callers can name the chunked scan explicitly."""
    return ssd_chunk_reference(*args, **kwargs)


__all__ = [
    "Mamba2Params",
    "Mamba2Result",
    "mamba2_chunked_reference",
    "mamba2_recurrent_reference",
    "mamba2_reference",
    "random_mamba2_params",
    "ssd_chunk_reference",
]
