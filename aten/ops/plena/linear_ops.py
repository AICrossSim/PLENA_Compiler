"""PLENA backend compatibility shims for linear operators."""


def linear_projection_plena(prog, input_var, weight_var, name: str = "linear_out"):
    return prog.linear_projection(input_var, weight_var, name)


def linear_plena(prog, input_var, weight_var, name: str = "linear_out"):
    return prog.linear_projection(input_var, weight_var, name)
