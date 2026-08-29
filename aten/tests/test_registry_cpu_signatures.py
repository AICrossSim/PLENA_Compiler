"""Every CPU backend in native_ops.yaml must accept the declared parameter list."""
import inspect, re, unittest
from compiler.aten.ops.registry import OpRegistry


class TestRegistryCpuSignatures(unittest.TestCase):
    def test_declared_arity_is_accepted_by_the_cpu_backend(self):
        # OpRegistry.dispatch forwards *args positionally with no binding -- the
        # `func:` string is used only to extract the op name -- so a CPU target whose
        # real signature differs from the declaration silently binds the wrong things.
        # mamba_gated_rmsnorm did exactly that: it pointed at rms_norm_cpu(input, eps,
        # ...), so the gate tensor bound to `eps` and the result was NaN.
        registry = OpRegistry.load()
        problems = []
        # Pre-existing deviations, recorded rather than fixed so that a NEW one fails
        # this test. flash_attention declares hq/hkv/h_qkv/batch_size/seq_len/kv_seq_len
        # for the PLENA backend's benefit; the CPU reference genuinely needs only
        # Q/K/V/scale, and every caller passes the rest by keyword, so nothing binds
        # wrongly today. Removing an entry from this set is a fix; adding one needs a
        # reason in the commit message.
        KNOWN_DEVIATIONS = {"flash_attention"}
        for name in registry.list_ops():
            schema = registry.get_op(name)
            declared = re.search(r"\((.*)\)\s*->", schema.func_signature)
            n_declared = 0 if not declared or not declared.group(1).strip() else len(
                declared.group(1).split(",")
            )
            fn = schema.resolve("cpu")
            params = list(inspect.signature(fn).parameters.values())
            n_positional = sum(
                1 for p in params
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            )
            if n_positional < n_declared and name not in KNOWN_DEVIATIONS:
                problems.append(
                    f"{name}: declares {n_declared} args but "
                    f"{fn.__module__}.{fn.__name__} takes {n_positional} positional"
                )
        self.assertEqual(problems, [], "\n".join(problems))


if __name__ == "__main__":
    unittest.main()
