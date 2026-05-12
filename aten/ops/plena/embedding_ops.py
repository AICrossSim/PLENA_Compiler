"""PLENA backend compatibility shims for positional encoding operators."""


def embedding_add_plena(prog, input_var, pos_weight_var):
    return prog.embedding_add(input_var, pos_weight_var)


def rope_plena(prog, x_var, x_rot_var, cos_var, sin_var):
    return prog.rope(x_var, x_rot_var, cos_var, sin_var)
