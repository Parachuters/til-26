import sys
import os
from pathlib import Path
from unittest import TestCase, main
from unittest.mock import patch

import numpy as np


AE_SRC = Path(__file__).resolve().parents[1] / "ae" / "src"
if str(AE_SRC) not in sys.path:
    sys.path.insert(0, str(AE_SRC))

from ae_manager import AEManager, FORWARD, PLACE_BOMB, RIGHT, STAY  # noqa: E402


def _observation(**overrides):
    obs = {
        "step": 1,
        "location": [4, 4],
        "base_location": [0, 0],
        "direction": 0,
        "health": [60.0],
        "base_health": [100.0],
        "team_resources": [0.0],
        "team_bombs": 1,
        "frozen_ticks": 0,
        "action_mask": [1, 1, 1, 1, 1, 1],
        "agent_viewcone": np.zeros((7, 5, 25), dtype=np.float32).tolist(),
        "base_viewcone": np.zeros((5, 5, 25), dtype=np.float32).tolist(),
    }
    obs.update(overrides)
    return obs


class AEManagerRuntimeTests(TestCase):
    def test_learned_policy_is_disabled_by_default(self):
        env = {key: value for key, value in os.environ.items() if key != "AE_ENABLE_RL"}
        with patch.dict(os.environ, env, clear=True):
            manager = AEManager()

        self.assertFalse(manager.learned_enabled)
        self.assertIsNone(manager.learned_policy)
        self.assertEqual(
            manager.source_counts,
            {
                "safety": 0,
                "planner": 0,
                "learned": 0,
                "fallback": 0,
            },
        )

    def test_forced_safety_runs_before_learned_policy(self):
        with patch.dict(os.environ, {"AE_ENABLE_RL": "1"}):
            manager = AEManager()
        manager.learned_policy = object()

        def unexpected_learned_action(*_args, **_kwargs):
            raise AssertionError("learned policy should not run during forced safety")

        manager._learned_action = unexpected_learned_action

        action = manager.ae(_observation(frozen_ticks=2))

        self.assertEqual(action, STAY)
        self.assertEqual(manager.source_counts["safety"], 1)
        self.assertEqual(manager.last_action_source, "safety")

    def test_high_confidence_planner_action_preempts_learned_policy(self):
        with patch.dict(os.environ, {"AE_ENABLE_RL": "1"}):
            manager = AEManager()
        manager.learned_policy = object()

        learned_calls = []
        manager._planner_priority_action = lambda mask, hazards: PLACE_BOMB
        manager._learned_action = (
            lambda observation, mask, hazards: learned_calls.append(observation) or FORWARD
        )

        action = manager.ae(_observation())

        self.assertEqual(action, PLACE_BOMB)
        self.assertEqual(learned_calls, [])
        self.assertEqual(manager.source_counts["planner"], 1)
        self.assertEqual(manager.last_action_source, "planner")

    def test_learned_policy_can_act_after_priority_gate(self):
        with patch.dict(os.environ, {"AE_ENABLE_RL": "1"}):
            manager = AEManager()
        manager.learned_policy = object()

        manager._planner_priority_action = lambda mask, hazards: None
        manager._learned_action = lambda observation, mask, hazards: RIGHT

        def unexpected_planner(*_args, **_kwargs):
            raise AssertionError("planner fallback should not run")

        manager._planner_action = unexpected_planner

        action = manager.ae(_observation())

        self.assertEqual(action, RIGHT)
        self.assertEqual(manager.source_counts["learned"], 1)
        self.assertEqual(manager.last_action_source, "learned")


if __name__ == "__main__":
    main()
