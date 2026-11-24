"""The `ablation` request field: validation (400s) and the GEN line's abl=/ablmask= keys."""
from __future__ import annotations

import unittest

from serve.config import checked_sampling
from serve.engine.ablation import ablation_key, checked_ablation
from serve.engine.process import ProcessEngine


class CheckedAblation(unittest.TestCase):
    def ok(self, value):
        checked_ablation(value)   # no raise

    def bad(self, value):
        with self.assertRaises(ValueError):
            checked_ablation(value)

    def test_strength(self):
        self.ok(0)
        self.ok(0.5)
        self.ok(1.0)
        self.bad(-0.1)
        self.bad(1.1)
        self.bad("0.5")
        self.bad(True)

    def test_object(self):
        self.ok({"strength": 0.8})
        self.ok({"strength": 1.0, "layers": [2, 3, 39]})
        self.bad({})
        self.bad({"strength": 2})
        self.bad({"strength": 0.5, "layers": []})
        self.bad({"strength": 0.5, "layers": [64]})
        self.bad({"strength": 0.5, "layers": [-1]})
        self.bad({"strength": 0.5, "layers": ["2"]})
        self.bad({"strength": 0.5, "weight": 1})

    def test_checked_sampling_raises(self):
        checked_sampling({})
        checked_sampling({"ablation": 0.5})
        with self.assertRaises(ValueError):
            checked_sampling({"ablation": 1.5})


class AblationKey(unittest.TestCase):
    def test_absent_and_zero_send_nothing(self):
        self.assertEqual(ablation_key({}), "")
        self.assertEqual(ablation_key({"ablation": 0}), "")
        self.assertEqual(ablation_key({"ablation": {"strength": 0, "layers": [2]}}), "")

    def test_strength_only(self):
        self.assertEqual(ablation_key({"ablation": 0.5}), " abl=0.5")
        self.assertEqual(ablation_key({"ablation": {"strength": 1.0}}), " abl=1.0")

    def test_layers_mask(self):
        self.assertEqual(ablation_key({"ablation": {"strength": 0.8, "layers": [2, 3]}}), " abl=0.8 ablmask=c")
        self.assertEqual(ablation_key({"ablation": {"strength": 0.8, "layers": [0]}}), " abl=0.8 ablmask=1")

    def test_gen_line_carries_the_keys(self):
        keys = ProcessEngine.sampling_keys({"temperature": 0.7, "ablation": {"strength": 0.5, "layers": [2, 39]}})
        self.assertIn("temperature=0.7", keys)
        self.assertIn(f"abl=0.5 ablmask={((1 << 2) | (1 << 39)):x}", keys)


if __name__ == "__main__":
    unittest.main()
