"""PLENA backend compatibility shim for FFN."""


def ffn_plena(prog, input_var, w_gate, w_up, w_down):
    return prog.ffn(input_var, w_gate, w_up, w_down)
