"""Regression: a plain Transformers model exposes set_adapter without an adapter."""
import unittest
import torch
from search_r1_team.model import HFPolicy

class FakeModel:
    def __init__(self):
        self.selected = None
        self.training = None
    def set_adapter(self, name):
        self.selected = name
    def train(self, mode):
        self.training = mode

class AdapterActivationTests(unittest.TestCase):
    def policy(self, has_adapters, training):
        policy = HFPolicy.__new__(HFPolicy)
        policy.lm = FakeModel()
        policy.has_adapters = has_adapters
        policy.training = training
        return policy

    def test_direct_model_does_not_select_nonexistent_adapter(self):
        policy = self.policy(False, False)
        def reject(name):
            raise ValueError("No adapter loaded")
        policy.lm.set_adapter = reject
        policy._activate("actor")
        self.assertFalse(policy.lm.training)

    def test_actor_adapter_selected_for_training(self):
        policy = self.policy(True, True)
        policy._activate("actor")
        self.assertEqual(policy.lm.selected, "actor")
        self.assertTrue(policy.lm.training)

    def test_critic_adapter_selected_for_training(self):
        policy = self.policy(True, True)
        policy._activate("critic")
        self.assertEqual(policy.lm.selected, "critic")

    def test_no_grad_rollout_is_eval_mode(self):
        policy = self.policy(True, True)
        with torch.no_grad():
            policy._activate("actor")
        self.assertFalse(policy.lm.training)

if __name__ == "__main__":
    unittest.main()
