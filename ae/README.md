# AE

Your AE challenge is to direct your agent through the game map while interacting with other agents and completing challenges.

This Readme provides a brief overview of the interface format; see the Wiki for the full [challenge specifications](https://github.com/til-ai/til-26/wiki/Challenge-specifications).

## Input

The input is sent via a POST request to the `/ae` route on port `5005`. It is a JSON object structured as such:

```JSON
{
  "instances": [
    {
      "observation": {
        "agent_viewcone": [[[0, ...], ...], ...],
        "base_viewcone":  [[[0, ...], ...], ...],
        "direction": 0,
        "location": [0, 0],
        "base_location": [0, 0],
        "health": [60.0],
        "frozen_ticks": 0,
        "base_health": [100.0],
        "team_resources": [0.0],
        "team_bombs": 0,
        "step": 0,
        "action_mask": [1, 1, 1, 1, 1, 0]
      }
    }
  ]
}
```

| Field | Shape / type | Description |
| --- | --- | --- |
| `agent_viewcone` | `float32 [7 × 5 × 25]` | Viewcone centred on this agent, oriented to its facing direction |
| `base_viewcone` | `float32 [5 × 5 × 25]` | Square view centred on the team base |
| `direction` | `Discrete(4)` | Facing direction (0=RIGHT, 1=DOWN, 2=LEFT, 3=UP) |
| `location` | `uint8 [2]` | Agent (x, y) grid position |
| `base_location` | `uint8 [2]` | Team base (x, y) grid position |
| `health` | `float32 [1]` | Agent current HP |
| `frozen_ticks` | `Discrete(freeze_turns+1)` | Remaining freeze steps (0 = active) |
| `base_health` | `float32 [1]` | Team base current HP |
| `team_resources` | `float32 [1]` | Accumulated resource ratio for this agent's team |
| `team_bombs` | `Discrete(max_team_bombs+1)` | Bomb stockpile for this agent's team |
| `step` | `Discrete(num_iters+1)` | Current step index |
| `action_mask` | `uint8 [6]` | Binary mask — 1 = action is legal this step |

The length of the `instances` array is 1.

NOTE: Reset as a POST endpoint on your ae_server.py will not be called. You are recommended to check if the current observation step is 0, and if so, reset your system internally. We will NOT be calling /reset to your server, for qualifiers OR finals.

## Output

Your route handler function must return a `dict` with this structure:

```Python
{
    "predictions": [
        {
            "action": 0
        }
    ]
}
```

The action is an integer:

| Index | Name | Description |
| --- | --- | --- |
| 0 | `FORWARD` | Move one cell in the facing direction |
| 1 | `BACKWARD` | Move one cell opposite to facing direction |
| 2 | `LEFT` | Turn 90° counter-clockwise |
| 3 | `RIGHT` | Turn 90° clockwise |
| 4 | `STAY` | Do not move |
| 5 | `PLACE_BOMB` | Place a bomb at the current cell (requires `team_bombs > 0`) |

## Current agent implementation

`src/ae_manager.py` implements a stateful planner-first hybrid agent for Advanced maps:

- Reconstructs a round-local belief map from `agent_viewcone` and `base_viewcone`.
- Infers base-view shape from the incoming tensor instead of hard-coding one size.
- Tracks seen walls, destructible walls, collectibles, enemy agents/bases, bombs, visited cells, and stale dynamic entities.
- Simulates bomb hazards over future timesteps and uses direction-aware BFS over `(x, y, direction, time)`.
- Prioritizes immediate bomb escape, high-confidence planner actions, optional learned policy actions, then planner fallback.
- Applies `action_mask` as the final guard and resets state on `step == 0`, step regression, empty `POST /ae`, or `POST /reset`.
- Keeps learned policy deployment opt-in: set `AE_ENABLE_RL=1` and `AE_POLICY_MODEL=ae/src/models/ae_policy.pt` only after local held-out evaluation beats planner-only.
- Logs action source counters for `safety`, `planner`, `learned`, and `fallback`.

## Evaluation plan

1. Run syntax and server smoke tests after each change:

```bash
python -m py_compile ae/src/ae_manager.py ae/src/ae_server.py
```

2. Build and run the official local AE test:

```bash
til build ae
til test ae
```

3. For faster iteration, run the direct test script while the AE container is already serving on port 5005:

```bash
python test/test_ae.py
```

4. Track at least these metrics across repeated six-round runs: total reward, invalid actions, frozen ticks/deaths, mission/resource/recon pickups, bomb damage, base damage dealt/taken, and mean response latency.

5. Validate with `TEAM_TRACK=advanced` and many random environment seeds. The local test uses random opponents, so also test against scripted greedy collectors and aggressive bombers before relying on the score.

## Planner-supervised training flow

1. Establish the rule-based baseline on the GCP Workbench instance:

```bash
til build ae
til test ae
```

Record `total rewards`, `score`, track, date, and commit before starting RL.

2. Collect planner demonstrations on Advanced random maps:

```bash
python ae/train/collect_planner_demos.py --episodes 2000 --seed-start 0
```

3. Train the recurrent masked behavior-cloning policy:

```bash
python ae/train/train_bc.py --demos ae/train/results/planner_demos.npz
```

Continue only if validation expert-action accuracy is high enough and BC rollout reward reaches the agreed floor.

4. Fine-tune from the BC checkpoint with masked recurrent PPO:

```bash
python ae/train/train_masked_ppo.py --init ae/train/checkpoints/ae_bc.pt
```

5. Compare planner, BC, PPO, and hybrid on identical held-out seeds:

```bash
python ae/train/evaluate_policies.py --episodes 60 --seed-start 20000
```

6. Deploy learned policy only if hybrid beats planner-only on mean reward, median reward, and lower-tail stability. Copy the winning compact checkpoint to `ae/src/models/ae_policy.pt`, add `torch` to `ae/requirements.txt` for that learned image, then run the container with:

```bash
AE_ENABLE_RL=1 AE_POLICY_MODEL=ae/src/models/ae_policy.pt
```

If the model, dependency, or confidence gate fails, `AEManager` automatically keeps using planner-only behavior.

7. Tune the heuristic weights in `TILE_WEIGHTS`, frontier scoring, bomb thresholds, and defense radius with random search or Bayesian optimization over the rollout harness.

8. Before submission, rebuild the CPU AE image, run `til test ae`, verify `GET /health`, and submit only the validated tag:

```bash
til build ae
til test ae
til submit ae
```
