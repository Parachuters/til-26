# AE (Autonomous Exploration) Solution Plan — Advanced Track

## Problem Summary

Navigate a **16×16 grid** maze for 200 timesteps, competing against 5 other teams. Complete objectives (collect resources, defend base, use bombs) to maximize reward.

**Scoring:** Sum of rewards ÷ rounds ÷ max score (1000)

**Advanced track difference:** Variable map layouts per match (cannot memorize map)

**Interface:** POST `/ae` port `5005`
- Input: observation dict (viewcone, location, health, resources, bombs, action_mask, etc.)
- Output: `{"action": int}` where int is 0–5

**Actions:**
| Index | Name |
|---|---|
| 0 | FORWARD |
| 1 | BACKWARD |
| 2 | LEFT (turn CCW) |
| 3 | RIGHT (turn CW) |
| 4 | STAY |
| 5 | PLACE_BOMB |

---

## Observation Space Breakdown

| Field | Shape | Description |
|---|---|---|
| `agent_viewcone` | `[7, 5, 25]` | 7-deep × 5-wide cone, 25 channels |
| `base_viewcone` | `[5, 5, 25]` in the current server/test payload | Square view from base, 25 channels; infer shape from the tensor where possible |
| `direction` | int 0-3 | 0=RIGHT, 1=DOWN, 2=LEFT, 3=UP |
| `location` | [x, y] | Agent position |
| `base_location` | [x, y] | Team base position |
| `health` | float | Agent HP (max likely 60.0) |
| `frozen_ticks` | int | Steps frozen (0 = active) |
| `base_health` | float | Base HP (max 100.0) |
| `team_resources` | float | Accumulated resource ratio |
| `team_bombs` | int | Bomb stockpile |
| `step` | int | Current timestep (0-199) |
| `action_mask` | [6] | 1 = action is legal |

---

## Strategy

### Phase 1 (Quick): Rule-Based Agent

Get a working, competitive agent fast before spending time on RL training.

### Phase 2 (Competition): RL-Trained Agent (PPO)

Train with `til_environment` and replace the rule-based logic.

---

## Phase 1: Rule-Based Agent

### State Tracking

```python
import numpy as np

class AEManager:
    def __init__(self):
        self.visited = set()
        self.map_knowledge = {}  # (x, y) -> cell type
        self.step = 0

    def ae(self, observation: dict) -> int:
        step = observation["step"]
        if step == 0:
            self._reset()
        return self._decide(observation)

    def _reset(self):
        self.visited = set()
        self.map_knowledge = {}
        self.step = 0

    def _decide(self, obs: dict) -> int:
        action_mask = obs["action_mask"]
        location = tuple(obs["location"])
        direction = obs["direction"]
        base_location = tuple(obs["base_location"])
        health = obs["health"][0]
        team_resources = obs["team_resources"][0]
        team_bombs = obs["team_bombs"]
        step = obs["step"]

        self.visited.add(location)

        # Priority 1: Don't stay frozen / use legal actions only
        legal = [i for i, m in enumerate(action_mask) if m == 1]

        # Priority 2: If low health, retreat toward base
        if health < 20.0:
            return self._move_toward(location, direction, base_location, legal)

        # Priority 3: Explore unvisited cells
        return self._explore(location, direction, legal)
```

### Navigation Helper

```python
    def _move_toward(self, loc, direction, target, legal):
        dx = target[0] - loc[0]
        dy = target[1] - loc[1]
        # Determine desired direction: 0=RIGHT (+x), 1=DOWN (+y), 2=LEFT (-x), 3=UP (-y)
        if abs(dx) >= abs(dy):
            desired_dir = 0 if dx > 0 else 2
        else:
            desired_dir = 1 if dy > 0 else 3

        if direction == desired_dir and 0 in legal:
            return 0  # FORWARD
        # Turn to face desired direction
        diff = (desired_dir - direction) % 4
        if diff == 1 and 3 in legal:
            return 3  # turn RIGHT
        if diff == 3 and 2 in legal:
            return 2  # turn LEFT
        if diff == 2 and 1 in legal:
            return 1  # BACKWARD
        return legal[0]  # fallback

    def _explore(self, loc, direction, legal):
        # Greedy: prefer unvisited forward cell
        forward_cell = self._forward_cell(loc, direction)
        if forward_cell not in self.visited and 0 in legal:
            return 0  # FORWARD
        # Rotate to explore
        if 3 in legal:
            return 3  # turn RIGHT to find new direction
        if 2 in legal:
            return 2
        if 0 in legal:
            return 0
        return legal[0]

    def _forward_cell(self, loc, direction):
        x, y = loc
        if direction == 0: return (x+1, y)
        if direction == 1: return (x, y+1)
        if direction == 2: return (x-1, y)
        if direction == 3: return (x, y-1)
```

### Viewcone Parsing

The 25 channels encode different cell features. Based on the `til_environment` documentation, channels likely include:
- Wall presence
- Resource locations
- Enemy agent positions
- Enemy base location
- Team member positions
- Bomb positions

```python
def parse_viewcone(viewcone: list) -> np.ndarray:
    # viewcone shape: [7, 5, 25] — depth=7, width=5, channels=25
    vc = np.array(viewcone)
    walls = vc[..., 0]       # channel 0: walls
    resources = vc[..., 1]   # channel 1: resources (verify from til_environment source)
    enemies = vc[..., 2]     # channel 2: enemies (verify)
    return walls, resources, enemies
```

Check `til-26-ae/` submodule source for exact channel definitions.

---

## Phase 2: RL Agent (PPO)

### Training Setup

```python
from stable_baselines3 import PPO
from til_environment import GridEnv  # from til-26-ae submodule

env = GridEnv(config={...})  # use advanced track config for variable maps

model = PPO(
    "MultiInputPolicy",
    env,
    verbose=1,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,        # encourage exploration
    tensorboard_log="./tb_logs/",
)
model.learn(total_timesteps=5_000_000)
model.save("ae_ppo")
```

### Policy Network

Use a custom CNN policy to process the viewcone:

```python
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class ViewconeExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(25, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Additional MLP for scalar features (health, step, etc.)
        self.mlp = nn.Linear(scalar_dim, 64)
        self.output = nn.Linear(cnn_out + 64, features_dim)
```

### Reward Shaping (Key for Training Quality)

The base reward is from challenge completions. Additional shaped rewards:
- +0.1 per new cell explored (exploration bonus, decay after step 100)
- -0.5 for being frozen
- +0.2 for collecting resources
- -1.0 for dying (health reaches 0)
- +reward for base defense (when base health is threatened)

### Variable Map Generalization

For advanced track (variable maps), ensure the agent generalizes by:
- Training on many different random maps (enable randomization in `GridEnv`)
- Using relative positions (from agent's current location) not absolute grid coordinates
- The viewcone is already relative to agent direction — this is the right input

---

## Inference Code (`ae_manager.py`)

```python
import numpy as np
from stable_baselines3 import PPO

class AEManager:
    def __init__(self):
        self.model = PPO.load("ae_ppo.zip")
        self.prev_step = -1

    def ae(self, observation: dict) -> int:
        step = observation["step"]
        if step == 0 and self.prev_step != 0:
            pass  # new round; model is stateless so nothing to reset
        self.prev_step = step

        # Ensure action_mask is respected
        action_mask = np.array(observation["action_mask"])
        
        obs = self._preprocess(observation)
        action, _ = self.model.predict(obs, deterministic=True)
        
        # Enforce action mask (safety)
        if action_mask[action] == 0:
            legal = np.where(action_mask == 1)[0]
            action = legal[0] if len(legal) > 0 else 4  # STAY as fallback
        
        return int(action)

    def _preprocess(self, observation: dict) -> dict:
        return {
            "agent_viewcone": np.array(observation["agent_viewcone"], dtype=np.float32),
            "base_viewcone": np.array(observation["base_viewcone"], dtype=np.float32),
            "scalars": np.array([
                observation["direction"],
                observation["location"][0] / 16.0,
                observation["location"][1] / 16.0,
                observation["health"][0] / 60.0,
                observation["frozen_ticks"] / 10.0,
                observation["base_health"][0] / 100.0,
                observation["team_resources"][0],
                observation["team_bombs"] / 10.0,
                observation["step"] / 200.0,
            ], dtype=np.float32),
        }
```

---

## Reset Behavior

Per the README: `/reset` will NOT be called by the competition system. Detect new round by checking `step == 0`:

```python
if observation["step"] == 0:
    self._reset_state()
```

---

## Dockerfile Notes

```dockerfile
FROM python:3.11-slim
RUN pip install stable-baselines3 gymnasium
COPY ae_ppo.zip /app/ae_ppo.zip
```

## requirements.txt

```
stable-baselines3>=2.3.0
gymnasium>=0.29.0
torch>=2.1.0
numpy
```

---

## Key Risks & Mitigations

| Risk | Mitigation |
|---|---|
| RL training takes too long | Start with rule-based agent; train RL in parallel |
| Variable maps cause poor generalization | Train on randomized maps; use relative observation inputs |
| Agent stays in place (entropy collapse) | Add entropy bonus (ent_coef=0.01) and exploration reward |
| Game rules not fully understood | Read til-26-ae source code carefully; check viewcone channels |
| /reset not called — stale state | Detect step==0 and reset internally |

---

## Scoring Checklist

- [ ] Rule-based agent working before RL training starts
- [ ] `action_mask` always enforced in output
- [ ] Reset triggered on `step == 0`
- [ ] Agent generalizes across different map layouts (test with random maps)
- [ ] Verify viewcone channel definitions from `til-26-ae` submodule
- [ ] Benchmark reward per round vs random baseline
