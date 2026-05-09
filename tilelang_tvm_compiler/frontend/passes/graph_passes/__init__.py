"""Graph-layer passes — operate on `Graph` (graph_ir.Graph), not on TIR
stmt trees. Each pass is a pure function ``Graph → Graph`` (or
``(Graph, scopes) → Graph`` if it needs scope info).

The migration plan is to gradually replace the stmt-walker passes
under ``frontend/passes/`` with graph-layer equivalents living here.
Phase 3.1 starts with ``annotate_sync``."""
