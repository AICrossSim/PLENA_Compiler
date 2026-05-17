"""Helper: should the cluster pipeline (pass_3 onwards) run on this MidFunc?

Two skip conditions, either alone is enough:

  1. ``MidFunc.lane_axes`` is empty — kernel didn't declare any axis
     for cluster fusion. Treat as "this kernel doesn't need cluster",
     no error.
  2. Every non-global buffer's last dim is already >= MLEN.
     A HW vector op is MLEN-wide; if every on-chip buffer already
     covers a full vector along its trailing axis, one lane fills
     a single instruction by itself — cluster fusion buys nothing.

The cluster passes (split / distribute_cluster / async_wrap / view /
fuse) all check this guard at entry and no-op if either condition
holds.
"""

from .ir import MidFunc


# MLEN: hardware vector width. Default for the current PLENA target.
# When per-target configuration is added, this should come from the
# target descriptor instead of being hard-coded.
MLEN = 64


def _last_dim(buf) -> int:
    if not buf.shape:
        return 0
    last = buf.shape[-1]
    return int(last) if isinstance(last, int) else 0


def should_skip_cluster(func: MidFunc) -> bool:
    """True if cluster fusion is unnecessary for this MidFunc."""
    if not func.lane_axes:
        return True
    non_global = [b for b in func.allocs if b.scope != "global"]
    if non_global and all(_last_dim(b) >= MLEN for b in non_global):
        return True
    return False


__all__ = ["should_skip_cluster", "MLEN"]
