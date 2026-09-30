import unittest
import compiler
import robust_space

class RobustBudgetTests(unittest.TestCase):
    def test_space_is_unique_and_exactly_budgeted(self):
        data=compiler.load_workloads()['workloads'][0]
        points=robust_space.designs()
        self.assertEqual(len({d['id'] for d in points}),len(points))
        self.assertEqual({tuple(d['lanes']) for d in points},set(compiler.ROBUST_ORGANIZATIONS))
        for d in points:
            p=compiler.compile_workload(data,d['lanes'],'whole',d['group'],d['resources'])
            b=p['budget'];h=p['hardware']
            self.assertEqual(sum(b['private_accumulator_bytes']),2*1024**2)
            self.assertEqual(sum(h['weight_slots']),10)
            self.assertEqual(sum(h['weight_banks']),64)
            self.assertEqual(sum(h['acc_banks']),2*sum(d['lanes']))
            self.assertEqual(b['x_total_bytes'],2048*sum(d['lanes']))
            self.assertLessEqual(p['controller']['total_state_bytes'],4096)
            for e in p['experts']:
                for v in e['candidates']:
                    for c in v['cores']:
                        self.assertEqual(c['storage']['feasible'], c['storage']['peak_private_bytes']<=c['storage']['budget_bytes'])
    def test_over_budget_and_unaligned_partitions_rejected(self):
        for field,value in [('weight_slots',[6,5]),('acc_banks',[8,5]),('acc_bytes',[1024**2+1,1024**2-1])]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                compiler.hardware_budget([4,2],{field:value})

    def test_joint_state_uses_existing_accumulator_capacity(self):
        data=compiler.load_workloads()['workloads'][0]
        for lanes in ([6],[3,3],[4,2],[8],[4,4],[5,3]):
            nc=len(lanes)
            base=dict(control_bytes=[4096//nc]*nc,feedback_state_bytes=96)
            before=compiler.compile_workload(data,lanes,'whole',4,base)
            resources=dict(base,joint_state_bytes=256,control_bytes=[4352//nc]*nc)
            after=compiler.compile_workload(data,lanes,'whole',4,resources)
            self.assertEqual(after['controller']['total_state_bytes']-before['controller']['total_state_bytes'],256)
            self.assertEqual(after['controller']['reserve_headroom_bytes'],before['controller']['reserve_headroom_bytes'])
            self.assertEqual(after['budget']['private_accumulator_total_bytes'],2*1024**2)
            self.assertEqual(after['budget']['control_reserve_inside_acc_bytes'],4352)
            self.assertEqual(sum(after['controller']['state_bytes_per_core']),after['controller']['total_state_bytes'])
        with self.assertRaises(ValueError):
            compiler.hardware_budget([4,2],dict(joint_state_bytes=256,control_bytes=[2048,2048]))
        with self.assertRaises(ValueError):
            compiler.hardware_budget([4,2],dict(joint_state_bytes=128,control_bytes=[2112,2112]))
