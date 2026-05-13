// precision.svh - PLENA numeric precision parameters.
//
// Intentionally minimal: generator/parser/hardware_parser.py (and the
// helpers under tools/) tolerate an empty precision file and fall back to the
// following per-parameter defaults when the corresponding `parameter` line is
// absent:
//
//   WT_MX_MANT_WIDTH   = 3,  WT_MX_EXP_WIDTH    = 4   // weight MX mantissa/exp
//   KV_MX_MANT_WIDTH   = 3,  KV_MX_EXP_WIDTH    = 4   // KV-cache MX mantissa/exp
//   ACT_MX_MANT_WIDTH  = 3,  ACT_MX_EXP_WIDTH   = 4   // activation MX mantissa/exp
//   MX_SCALE_WIDTH     = 3,  SCALE_MX_EXP_WIDTH = 4   // shared-scale mantissa/exp
//   BLOCK_DIM          = 4                            // MX block dimension
//
// Override any of them by declaring, e.g.:
//
//   parameter WT_MX_MANT_WIDTH = 4;
//
// Included transitively from configuration.svh via:
//   `include "precision.svh"
//
// This header is the canonical source-of-truth; the compiler reads it directly
// from compiler/doc/ and the RTL's configuration.svh includes it from the same
// directory.
`ifndef PRECISION_SVH
`define PRECISION_SVH

package precision_pkg;
endpackage

`endif
