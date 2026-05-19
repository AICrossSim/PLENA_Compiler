"""PLENA backend wrapper for Flash Attention."""


def flash_attention_plena(
    prog,
    Q,
    K,
    V,
    scale=None,
    hq=1,
    hkv=1,
    h_qkv=None,
    causal_mask=None,
    batch_size=1,
    seq_len=None,
    kv_seq_len=None,
):
    return prog.flash_attention(
        Q,
        K,
        V,
        scale=scale,
        hq=hq,
        hkv=hkv,
        h_qkv=h_qkv,
        causal_mask=causal_mask,
        batch_size=batch_size,
        seq_len=seq_len,
        kv_seq_len=kv_seq_len,
    )
