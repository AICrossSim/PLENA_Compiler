"""Predeclared finite physical search space; independent of workload timings."""
from itertools import product
import compiler


def designs():
    result = []
    for lanes in compiler.ROBUST_ORGANIZATIONS:
        nc, total = len(lanes), sum(lanes)
        slot_splits = [(10,)] if nc == 1 else [(i, 10-i) for i in range(3,8)]
        seen = set()
        for slots, arena, banks, group in product(slot_splits, ('proportional','equal'), ('proportional','equal'), (2,4)):
            if min(slots)<=group:
                continue
            if lanes[0] == lanes[-1] and nc == 2 and slots[0] < slots[1]:
                continue  # Mirror-identical homogeneous hardware; label core0 with more slots.
            shares = lanes if arena == 'proportional' else [1]*nc
            port_shares = lanes if banks == 'proportional' else [1]*nc
            resources = dict(weight_slots=list(slots), weight_banks=[64//nc]*nc,
                             x_banks=[4*m for m in lanes],
                             acc_banks=compiler.partition_bytes(2*total, list(port_shares), 1),
                             acc_bytes=compiler.partition_bytes(compiler.ACC_BYTES, list(shares)),
                             control_bytes=[4096//nc]*nc, feedback_state_bytes=96)
            hardware = compiler.hardware_budget(lanes, resources)
            identity = (group,) + tuple(tuple(hardware[k]) if isinstance(hardware[k],list) else hardware[k]
                             for k in sorted(hardware))
            if identity in seen:
                continue
            seen.add(identity)
            org = '+'.join(map(str,lanes))
            result.append(dict(id=f'M{total}_{org}_W{"-".join(map(str,slots))}_A{arena}_P{banks}_G{group}',
                               budget_group=f'M{total}', lanes=list(lanes), group=group,
                               architecture='single' if nc==1 else 'homogeneous' if len(set(lanes))==1 else 'heterogeneous',
                               resources=resources, arena_policy=arena, bank_policy=banks))
    return result
