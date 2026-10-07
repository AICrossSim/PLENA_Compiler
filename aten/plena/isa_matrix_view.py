"""Compatibility import for the single maintained Matrix-view contract.

Older callers use ``isa_matrix_view``; newer projection/L-TILE emitters use
``mview``. Both must share the same descriptor classes and primitive enum so a
validated view can cross the public projection API without nominal-type errors.
The previous contract is preserved in the round2 research archive/history.
"""
from compiler.aten.plena.mview import *  # noqa: F401,F403
from compiler.aten.plena.mview import __all__
