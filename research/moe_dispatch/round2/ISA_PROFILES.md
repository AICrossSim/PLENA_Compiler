# Authoritative and archived ISA profiles

The active profile is `doc/operation.svh`, inventoried in `active_isa_profile.json`. Existing ordinary instructions and latest Matrix views/LSTREAM/L-TILE remain enabled. Routed control uses 0x39--0x3C. General recurrent operations use V_SOFTPLUS_V=0x3D, S_MAP_FP_V=0x3E and L_TILE=0x3F.

These assignments are incompatible with older prototypes: X_MAMBA at 0x39; X_STATE at 0x3D; L_SCATTER_M at 0x3F; and an alternative consolidated C_ROUTE dispatcher. Their branch histories and original full conflicts are archived under the corresponding branch directory. No opcode was arbitrarily relocated to claim simultaneous support.

TOPK extends its existing policy register, without spending an extra opcode: top_k[7:0], expert_count[21:8], sigmoid-normalized flag[22], correction-bias flag[23]. C_SET_TOPK_REG target 0 writes policy; target 1 writes the correction-bias VRAM base. The one-operand spelling remains target 0 and byte-identical.

Historical codec/descriptor unit tests can still validate their original isolated byte layouts. Tests that assemble an unavailable prototype through the active opcode map explicitly report retired-profile skips. Original test versions are preserved under `archive/retired_isa_tests`; modern tests assert the active contract.
