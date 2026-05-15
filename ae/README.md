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

`src/ae_manager.py` implements a stateful hybrid planner for Advanced maps:

- Reconstructs a round-local belief map from `agent_viewcone` and `base_viewcone`.
- Infers base-view shape from the incoming tensor instead of hard-coding one size.
- Tracks seen walls, destructible walls, collectibles, enemy agents/bases, bombs, visited cells, and stale dynamic entities.
- Simulates bomb hazards over future timesteps and uses direction-aware BFS over `(x, y, direction, time)`.
- Prioritizes immediate bomb escape, safe opportunistic bombing, nearby base defense, reward collection, frontier exploration, and useful wall bombing.
- Applies `action_mask` as the final guard and resets state on `step == 0`, step regression, empty `POST /ae`, or `POST /reset`.

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

## Training and deployment follow-ups

1. Add a local rollout harness that imports `til_environment` directly, runs hundreds of Advanced seeds, and logs reward breakdowns plus action/error counters to CSV.

2. Tune the heuristic weights in `TILE_WEIGHTS`, frontier scoring, bomb thresholds, and defense radius with random search or Bayesian optimization over the rollout harness.

3. Keep the planner as the safety layer. If learning is added, start with a small learned goal scorer that ranks candidate targets; do not let a neural policy bypass bomb-escape and action-mask checks.

4. Before submission, rebuild the CPU AE image, run `til test ae`, verify `GET /health`, and submit only the validated tag:

```bash
til build ae
til test ae
til submit ae
```
