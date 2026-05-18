"""Stateful hybrid planner for the AE Bomberman challenge."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import inf
import os
from pathlib import Path
from typing import Iterable

import numpy as np

# View channels.
VISIBLE = 0
WALL_RIGHT = 1
WALL_DOWN = 2
WALL_LEFT = 3
WALL_UP = 4
TILE_EMPTY = 5
TILE_RECON = 6
TILE_MISSION = 7
TILE_RESOURCE = 8
ALLY_AGENT = 9
ENEMY_AGENT = 10
ALLY_BASE = 11
ENEMY_BASE = 12
DESTR_WALL_RIGHT = 13
DESTR_WALL_DOWN = 14
DESTR_WALL_LEFT = 15
DESTR_WALL_UP = 16
ALLY_BOMB = 17
ENEMY_BOMB = 18
ALLY_BOMB_TIMER = 19
ENEMY_BOMB_TIMER = 20
ALLY_BASE_HEALTH = 23

WALL_CH = (WALL_RIGHT, WALL_DOWN, WALL_LEFT, WALL_UP)
DESTR_CH = (DESTR_WALL_RIGHT, DESTR_WALL_DOWN, DESTR_WALL_LEFT, DESTR_WALL_UP)

# Actions.
FORWARD = 0
BACKWARD = 1
LEFT = 2
RIGHT = 3
STAY = 4
PLACE_BOMB = 5

# Direction order matches til_environment.types.Direction.
DIRS = ((1, 0), (0, 1), (-1, 0), (0, -1))

GRID_SIZE = 16
BOMB_RADIUS = 2
BOMB_TIMER_AFTER_PLACEMENT = 4
MAX_PLAN_DEPTH = 36
DYNAMIC_ENTITY_TTL = 8

TILE_WEIGHTS = {
    "mission": 22.0,
    "resource": 13.0,
    "recon": 5.0,
}


Cell = tuple[int, int]
Edge = tuple[Cell, Cell]


@dataclass
class SeenEntity:
    pos: Cell
    last_seen: int
    health_ratio: float = 1.0


@dataclass
class BombInfo:
    pos: Cell
    team: str
    timer: int
    last_seen: int


@dataclass
class BeliefState:
    step: int = -1
    pos: Cell = (0, 0)
    direction: int = 0
    base_pos: Cell = (0, 0)
    base_health: float = 100.0
    known_cells: set[Cell] = field(default_factory=set)
    known_edges: set[Edge] = field(default_factory=set)
    blocked_edges: set[Edge] = field(default_factory=set)
    destructible_edges: set[Edge] = field(default_factory=set)
    tiles: dict[Cell, str] = field(default_factory=dict)
    enemy_agents: dict[Cell, SeenEntity] = field(default_factory=dict)
    enemy_bases: dict[Cell, SeenEntity] = field(default_factory=dict)
    bombs: dict[tuple[Cell, str], BombInfo] = field(default_factory=dict)
    visited_count: np.ndarray = field(
        default_factory=lambda: np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int16)
    )
    last_seen: np.ndarray = field(
        default_factory=lambda: np.full((GRID_SIZE, GRID_SIZE), -999, dtype=np.int16)
    )


class AEManager:
    """Fast deterministic agent with online map memory and safe planning."""

    def __init__(self) -> None:
        self.state = BeliefState()
        self.source_counts = {
            "safety": 0,
            "planner": 0,
            "learned": 0,
            "fallback": 0,
        }
        self.last_action_source: str | None = None
        self.learned_enabled = os.getenv("AE_ENABLE_RL", "").lower() in {
            "1",
            "true",
            "yes",
        }
        self.learned_min_confidence = float(os.getenv("AE_RL_MIN_CONFIDENCE", "0.55"))
        self.learned_hidden_state = None
        self.learned_policy = self._load_learned_policy()
        self.rl_model = self.learned_policy

        if not self.learned_enabled:
            print("AE learned policy disabled; using planner-only mode.")
        elif self.learned_policy is not None:
            print("AE learned policy loaded; hybrid gate is active.")
        else:
            print("AE learned policy requested but unavailable; using planner-only mode.")

    def reset(self) -> None:
        self.state = BeliefState()
        self.learned_hidden_state = None

    def ae(self, observation: dict) -> int:
        if not observation:
            self.reset()
            return STAY

        step = int(observation.get("step", 0))
        if step == 0 or step < self.state.step:
            self.reset()

        mask = np.asarray(observation.get("action_mask", []), dtype=np.int8)
        if mask.size < 6 or not np.any(mask):
            return STAY

        self._update_state(observation)
        hazards = self._build_hazards()

        forced_action = self._forced_safety_action(observation, mask, hazards)
        if forced_action is not None:
            self._count_source("safety")
            return forced_action

        priority_action = self._planner_priority_action(mask, hazards)
        if priority_action is not None:
            self._count_source("planner")
            return priority_action

        learned_action = self._learned_action(observation, mask, hazards)
        if learned_action is not None:
            self._count_source("learned")
            return learned_action

        self._count_source("fallback")
        return self._planner_action(mask, hazards)

    def _count_source(self, source: str) -> None:
        self.last_action_source = source
        self.source_counts[source] = self.source_counts.get(source, 0) + 1
        total = sum(self.source_counts.values())
        if total > 0 and total % 100 == 0:
            print(f"AE action source counts: {self.source_counts}")

    def _load_learned_policy(self) -> object | None:
        if not self.learned_enabled:
            return None

        default_path = Path(__file__).resolve().parent / "models" / "ae_policy.pt"
        model_path = Path(os.getenv("AE_POLICY_MODEL", str(default_path)))
        if not model_path.exists():
            return None

        try:
            from masked_recurrent_policy import load_masked_recurrent_policy

            return load_masked_recurrent_policy(model_path, device="cpu")
        except Exception as exc:
            import traceback

            print(f"AE learned policy load failed: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            return None

    def _forced_safety_action(
        self,
        observation: dict,
        mask: np.ndarray,
        hazards: dict[int, set[Cell]],
    ) -> int | None:
        if int(observation.get("frozen_ticks", 0)) > 0:
            return self._legal_or_fallback(STAY, mask, hazards)

        if self._cell_danger(self.state.pos, hazards, 1) or self._cell_danger(
            self.state.pos, hazards, 2
        ):
            action = self._path_to_nearest_safe(hazards)
            return self._legal_or_fallback(action, mask, hazards)

        return None

    def _planner_priority_action(
        self, mask: np.ndarray, hazards: dict[int, set[Cell]]
    ) -> int | None:
        if self._should_place_attack_bomb(mask, hazards):
            return PLACE_BOMB

        defense_target = self._base_defense_target()
        if defense_target is not None:
            action = self._plan_first_action(defense_target, hazards, max_depth=20)
            if action is not None:
                return self._legal_or_fallback(action, mask, hazards)

        ranked = self._ranked_targets(hazards)
        if ranked:
            target, score = ranked[0]
            close_high_value = self._manhattan(self.state.pos, target) <= 2 and score >= 10.0
            if score >= float(os.getenv("AE_PLANNER_PRIORITY_SCORE", "18.0")) or close_high_value:
                action = self._plan_first_action(target, hazards)
                if action is not None:
                    return self._legal_or_fallback(action, mask, hazards)

        if self._wall_bomb_value() >= 10.0 and self._should_bomb_wall(mask, hazards):
            return PLACE_BOMB

        return None

    def _learned_action(
        self,
        observation: dict,
        mask: np.ndarray,
        hazards: dict[int, set[Cell]],
    ) -> int | None:
        if self.learned_policy is None:
            return None

        try:
            result = self.learned_policy.act(
                observation,
                hidden_state=self.learned_hidden_state,
                deterministic=True,
            )
            candidate = int(result.action)
            confidence = float(result.confidence)
            self.learned_hidden_state = result.hidden_state
        except Exception:
            return None

        if confidence < self.learned_min_confidence:
            return None

        if candidate == PLACE_BOMB and not self._escape_after_bomb_exists(hazards):
            return None

        filtered = self._legal_or_fallback(candidate, mask, hazards)
        if filtered == candidate:
            return filtered
        return None

    def _rl_action(
        self,
        observation: dict,
        mask: np.ndarray,
        hazards: dict[int, set[Cell]],
    ) -> int | None:
        return self._learned_action(observation, mask, hazards)

    def _planner_action(self, mask: np.ndarray, hazards: dict[int, set[Cell]]) -> int:
        if self._should_place_attack_bomb(mask, hazards):
            return PLACE_BOMB

        defense_target = self._base_defense_target()
        if defense_target is not None:
            action = self._plan_first_action(defense_target, hazards, max_depth=20)
            if action is not None:
                return self._legal_or_fallback(action, mask, hazards)

        for target, _score in self._ranked_targets(hazards):
            action = self._plan_first_action(target, hazards)
            if action is not None:
                return self._legal_or_fallback(action, mask, hazards)

        if self._should_bomb_wall(mask, hazards):
            return PLACE_BOMB

        return self._legal_or_fallback(self._explore_action(hazards), mask, hazards)

    # ------------------------------------------------------------------
    # Belief updates
    # ------------------------------------------------------------------

    def _update_state(self, observation: dict) -> None:
        state = self.state
        step = int(observation.get("step", 0))
        state.step = step
        state.pos = self._cell(observation.get("location", [0, 0]))
        state.direction = int(observation.get("direction", 0)) % 4
        state.base_pos = self._cell(observation.get("base_location", [0, 0]))
        state.base_health = self._scalar(observation.get("base_health", [100.0]), 100.0)

        if self._in_bounds(state.pos):
            state.visited_count[state.pos] += 1

        agent_view = np.asarray(observation.get("agent_viewcone", []), dtype=np.float32)
        base_view = np.asarray(observation.get("base_viewcone", []), dtype=np.float32)

        if agent_view.ndim == 3 and agent_view.shape[-1] >= 25:
            origin = self._agent_origin(agent_view)
            for r in range(agent_view.shape[0]):
                for c in range(agent_view.shape[1]):
                    cell = self._agent_view_to_world((r, c), origin)
                    self._update_visible_cell(cell, agent_view[r, c])

        if base_view.ndim == 3 and base_view.shape[-1] >= 25:
            origin = (base_view.shape[0] // 2, base_view.shape[1] // 2)
            for r in range(base_view.shape[0]):
                for c in range(base_view.shape[1]):
                    x = state.base_pos[0] + r - origin[0]
                    y = state.base_pos[1] + c - origin[1]
                    self._update_visible_cell((x, y), base_view[r, c])

        self._decay_dynamic_memory()

    def _update_visible_cell(self, cell: Cell, ch: np.ndarray) -> None:
        if not self._in_bounds(cell) or ch[VISIBLE] <= 0.0:
            return

        state = self.state
        x, y = cell
        state.known_cells.add(cell)
        state.last_seen[x, y] = state.step

        for direction in range(4):
            edge = self._edge(cell, direction)
            if edge is None:
                continue
            wall_present = ch[WALL_CH[direction]] > 0.5
            wall_destructible = ch[DESTR_CH[direction]] > 0.5
            state.known_edges.add(edge)
            if wall_present:
                state.blocked_edges.add(edge)
                if wall_destructible:
                    state.destructible_edges.add(edge)
                else:
                    state.destructible_edges.discard(edge)
            else:
                state.blocked_edges.discard(edge)
                state.destructible_edges.discard(edge)

        if ch[TILE_MISSION] > 0.5:
            state.tiles[cell] = "mission"
        elif ch[TILE_RESOURCE] > 0.5:
            state.tiles[cell] = "resource"
        elif ch[TILE_RECON] > 0.5:
            state.tiles[cell] = "recon"
        elif ch[TILE_EMPTY] > 0.5:
            state.tiles.pop(cell, None)

        if ch[ENEMY_AGENT] > 0.5:
            state.enemy_agents[cell] = SeenEntity(cell, state.step, float(ch[22]))
        else:
            state.enemy_agents.pop(cell, None)

        if ch[ENEMY_BASE] > 0.5:
            health = float(ch[24]) if ch[24] > 0 else 1.0
            state.enemy_bases[cell] = SeenEntity(cell, state.step, health)
        else:
            state.enemy_bases.pop(cell, None)

        for key in [key for key in state.bombs if key[0] == cell]:
            state.bombs.pop(key, None)
        if ch[ALLY_BOMB] > 0.5:
            state.bombs[(cell, "ally")] = BombInfo(
                cell, "ally", int(round(float(ch[ALLY_BOMB_TIMER]))), state.step
            )
        if ch[ENEMY_BOMB] > 0.5:
            state.bombs[(cell, "enemy")] = BombInfo(
                cell, "enemy", int(round(float(ch[ENEMY_BOMB_TIMER]))), state.step
            )

    def _decay_dynamic_memory(self) -> None:
        step = self.state.step
        self.state.enemy_agents = {
            cell: ent
            for cell, ent in self.state.enemy_agents.items()
            if step - ent.last_seen <= DYNAMIC_ENTITY_TTL
        }
        self.state.bombs = {
            key: bomb
            for key, bomb in self.state.bombs.items()
            if self._predicted_bomb_timer(bomb) >= -1
        }

    # ------------------------------------------------------------------
    # Hazards and bombing
    # ------------------------------------------------------------------

    def _build_hazards(
        self, extra_bombs: Iterable[BombInfo] = ()
    ) -> dict[int, set[Cell]]:
        hazards: dict[int, set[Cell]] = {}
        for bomb in [*self.state.bombs.values(), *extra_bombs]:
            timer = self._predicted_bomb_timer(bomb)
            explode_at = max(0, timer) + 1
            cells = self._blast_cells(bomb.pos)
            hazards.setdefault(explode_at, set()).update(cells)
            hazards.setdefault(explode_at + 1, set()).update(cells)
        return hazards

    def _predicted_bomb_timer(self, bomb: BombInfo) -> int:
        return int(bomb.timer) - max(0, self.state.step - bomb.last_seen)

    def _blast_cells(self, origin: Cell) -> set[Cell]:
        cells: set[Cell] = set()
        ox, oy = origin
        for x in range(max(0, ox - BOMB_RADIUS), min(GRID_SIZE, ox + BOMB_RADIUS + 1)):
            for y in range(
                max(0, oy - BOMB_RADIUS), min(GRID_SIZE, oy + BOMB_RADIUS + 1)
            ):
                if self._line_clear(origin, (x, y)):
                    cells.add((x, y))
        return cells

    def _should_place_attack_bomb(
        self, mask: np.ndarray, hazards: dict[int, set[Cell]]
    ) -> bool:
        if not bool(mask[PLACE_BOMB]):
            return False

        blast = self._blast_cells(self.state.pos)
        value = 0.0
        for cell, base in self.state.enemy_bases.items():
            if cell in blast:
                value += 45.0 + (1.0 - base.health_ratio) * 20.0
        for cell, agent in self.state.enemy_agents.items():
            if cell in blast and self.state.step - agent.last_seen <= 2:
                value += 20.0 + (1.0 - agent.health_ratio) * 10.0
        if self.state.base_pos in blast:
            value -= 200.0

        return value >= 20.0 and self._escape_after_bomb_exists(hazards)

    def _should_bomb_wall(self, mask: np.ndarray, hazards: dict[int, set[Cell]]) -> bool:
        if not bool(mask[PLACE_BOMB]):
            return False
        if self._wall_bomb_value() < 7.0:
            return False
        return self._escape_after_bomb_exists(hazards)

    def _wall_bomb_value(self) -> float:
        blast = self._blast_cells(self.state.pos)
        value = 0.0
        for edge in self.state.destructible_edges:
            a, b = edge
            if a not in blast and b not in blast:
                continue
            unknown = sum(1 for cell in edge if cell not in self.state.known_cells)
            visits = sum(
                int(self.state.visited_count[cell])
                for cell in edge
                if self._in_bounds(cell)
            )
            value += 2.5 + unknown * 3.0 - min(2.0, visits * 0.2)
        return value

    def _escape_after_bomb_exists(self, hazards: dict[int, set[Cell]]) -> bool:
        own_bomb = BombInfo(
            self.state.pos,
            "ally",
            BOMB_TIMER_AFTER_PLACEMENT,
            self.state.step,
        )
        future_hazards = self._build_hazards([own_bomb])
        for t, cells in hazards.items():
            future_hazards.setdefault(t, set()).update(cells)

        start = (self.state.pos, self.state.direction, 1)
        queue = deque([start])
        seen = {start}
        while queue:
            pos, direction, time = queue.popleft()
            if time >= BOMB_TIMER_AFTER_PLACEMENT + 1:
                if not self._cell_danger(pos, future_hazards, time):
                    return True
                continue
            for action in (FORWARD, BACKWARD, LEFT, RIGHT, STAY):
                nxt = self._transition(pos, direction, action)
                if nxt is None:
                    continue
                npos, ndir = nxt
                ntime = time + 1
                state = (npos, ndir, ntime)
                if state in seen or self._cell_danger(npos, future_hazards, ntime):
                    continue
                seen.add(state)
                queue.append(state)
        return False

    # ------------------------------------------------------------------
    # Target selection
    # ------------------------------------------------------------------

    def _base_defense_target(self) -> Cell | None:
        base = self.state.base_pos
        if self._manhattan(self.state.pos, base) > 8:
            return None
        recent = [
            ent.pos
            for ent in self.state.enemy_agents.values()
            if self.state.step - ent.last_seen <= 3 and self._manhattan(ent.pos, base) <= 5
        ]
        if recent:
            return min(recent, key=lambda cell: self._manhattan(self.state.pos, cell))
        return None

    def _ranked_targets(self, hazards: dict[int, set[Cell]]) -> list[tuple[Cell, float]]:
        del hazards  # reserved for future learned/risk-aware scoring
        targets: dict[Cell, float] = {}

        for cell, kind in self.state.tiles.items():
            if not self._in_bounds(cell):
                continue
            age = self.state.step - int(self.state.last_seen[cell])
            weight = TILE_WEIGHTS.get(kind, 0.0)
            score = (
                weight
                - 0.42 * self._manhattan(self.state.pos, cell)
                - 0.55 * int(self.state.visited_count[cell])
                - 0.03 * max(0, age)
            )
            targets[cell] = max(targets.get(cell, -inf), score)

        for cell, ent in self.state.enemy_bases.items():
            age = self.state.step - ent.last_seen
            score = 34.0 - 0.35 * self._manhattan(self.state.pos, cell) - age * 0.2
            targets[cell] = max(targets.get(cell, -inf), score)

        for cell, ent in self.state.enemy_agents.items():
            if self.state.step - ent.last_seen > 3:
                continue
            score = 10.0 - 0.55 * self._manhattan(self.state.pos, cell)
            targets[cell] = max(targets.get(cell, -inf), score)

        for cell in self._frontier_cells():
            unseen = self._unknown_neighbor_count(cell)
            center_bonus = 2.0 - 0.15 * self._manhattan(cell, (7, 7))
            score = (
                3.2 * unseen
                + center_bonus
                - 0.32 * self._manhattan(self.state.pos, cell)
                - 0.8 * int(self.state.visited_count[cell])
            )
            targets[cell] = max(targets.get(cell, -inf), score)

        ranked = sorted(targets.items(), key=lambda item: item[1], reverse=True)
        return ranked[:30]

    def _frontier_cells(self) -> list[Cell]:
        frontiers: list[Cell] = []
        for cell in self.state.known_cells:
            if not self._in_bounds(cell):
                continue
            if self._unknown_neighbor_count(cell) > 0:
                frontiers.append(cell)
        if frontiers:
            return frontiers

        # Fallback for the first few steps when little is known.
        candidates = []
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                if self.state.visited_count[x, y] == 0:
                    candidates.append((x, y))
        return sorted(candidates, key=lambda c: self._manhattan(self.state.pos, c))[:12]

    def _unknown_neighbor_count(self, cell: Cell) -> int:
        count = 0
        for direction in range(4):
            nxt = self._move(cell, direction)
            if self._in_bounds(nxt) and nxt not in self.state.known_cells:
                count += 1
        return count

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def _plan_first_action(
        self,
        target: Cell,
        hazards: dict[int, set[Cell]],
        max_depth: int = MAX_PLAN_DEPTH,
    ) -> int | None:
        if target == self.state.pos:
            return STAY

        start = (self.state.pos, self.state.direction, 0)
        queue = deque([(start, None)])
        seen = {start}

        while queue:
            (pos, direction, time), first_action = queue.popleft()
            if time >= max_depth:
                continue

            for action in (FORWARD, BACKWARD, LEFT, RIGHT, STAY):
                nxt = self._transition(pos, direction, action)
                if nxt is None:
                    continue
                npos, ndir = nxt
                ntime = time + 1
                if self._cell_danger(npos, hazards, ntime):
                    continue
                state = (npos, ndir, ntime)
                if state in seen:
                    continue
                next_first = action if first_action is None else first_action
                if npos == target:
                    return next_first
                seen.add(state)
                queue.append((state, next_first))
        return None

    def _path_to_nearest_safe(self, hazards: dict[int, set[Cell]]) -> int | None:
        start = (self.state.pos, self.state.direction, 0)
        queue = deque([(start, None)])
        seen = {start}

        while queue:
            (pos, direction, time), first_action = queue.popleft()
            if time > 0 and not self._cell_danger(pos, hazards, time):
                return first_action if first_action is not None else STAY
            if time >= 12:
                continue

            for action in (FORWARD, BACKWARD, LEFT, RIGHT, STAY):
                nxt = self._transition(pos, direction, action)
                if nxt is None:
                    continue
                npos, ndir = nxt
                ntime = time + 1
                if self._cell_danger(npos, hazards, ntime):
                    continue
                state = (npos, ndir, ntime)
                if state in seen:
                    continue
                seen.add(state)
                queue.append((state, action if first_action is None else first_action))
        return None

    def _explore_action(self, hazards: dict[int, set[Cell]]) -> int:
        best_action = STAY
        best_score = -inf
        for action in (FORWARD, BACKWARD, RIGHT, LEFT, STAY):
            nxt = self._transition(self.state.pos, self.state.direction, action)
            if nxt is None:
                continue
            npos, _ndir = nxt
            if self._cell_danger(npos, hazards, 1):
                continue
            score = (
                2.0 * self._unknown_neighbor_count(npos)
                - 0.7 * int(self.state.visited_count[npos])
                - (0.2 if action == STAY else 0.0)
            )
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    def _transition(self, pos: Cell, direction: int, action: int) -> tuple[Cell, int] | None:
        if action == LEFT:
            return pos, (direction - 1) % 4
        if action == RIGHT:
            return pos, (direction + 1) % 4
        if action == STAY or action == PLACE_BOMB:
            return pos, direction

        move_dir = direction if action == FORWARD else (direction + 2) % 4
        nxt = self._move(pos, move_dir)
        if not self._in_bounds(nxt) or self._blocked(pos, move_dir):
            return None
        return nxt, direction

    def _legal_or_fallback(
        self, action: int | None, mask: np.ndarray, hazards: dict[int, set[Cell]]
    ) -> int:
        legal = [idx for idx, value in enumerate(mask[:6]) if value]
        if not legal:
            return STAY

        if action is not None and 0 <= action < 6 and bool(mask[action]):
            if action == PLACE_BOMB:
                return action
            nxt = self._transition(self.state.pos, self.state.direction, action)
            if nxt is not None and not self._cell_danger(nxt[0], hazards, 1):
                return action

        for candidate in (FORWARD, BACKWARD, RIGHT, LEFT, STAY):
            if candidate not in legal:
                continue
            nxt = self._transition(self.state.pos, self.state.direction, candidate)
            if nxt is not None and not self._cell_danger(nxt[0], hazards, 1):
                return candidate
        return int(legal[0])

    # ------------------------------------------------------------------
    # Geometry and tensor mapping
    # ------------------------------------------------------------------

    def _agent_origin(self, view: np.ndarray) -> tuple[int, int]:
        default = (min(2, view.shape[0] - 1), view.shape[1] // 2)
        allies = np.argwhere((view[..., ALLY_AGENT] > 0.5) & (view[..., VISIBLE] > 0.5))
        if allies.size == 0:
            return default
        return tuple(
            min(
                ((int(r), int(c)) for r, c in allies),
                key=lambda rc: abs(rc[0] - default[0]) + abs(rc[1] - default[1]),
            )
        )

    def _agent_view_to_world(self, idx: tuple[int, int], origin: tuple[int, int]) -> Cell:
        local = (idx[0] - origin[0], idx[1] - origin[1])
        x, y = self.state.pos
        row, col = local
        direction = self.state.direction
        if direction == 0:
            return x + row, y + col
        if direction == 1:
            return x - col, y + row
        if direction == 2:
            return x - row, y - col
        return x + col, y - row

    def _line_clear(self, start: Cell, end: Cell) -> bool:
        if start == end:
            return True
        path = self._supercover_line(start, end)
        for curr, nxt in zip(path, path[1:]):
            dx = nxt[0] - curr[0]
            dy = nxt[1] - curr[1]
            if dx != 0 and dy != 0:
                horizontal = self._blocked(curr, 0 if dx > 0 else 2)
                vertical = self._blocked(curr, 1 if dy > 0 else 3)
                if horizontal and vertical:
                    return False
            else:
                direction = 0 if dx > 0 else 2 if dx < 0 else 1 if dy > 0 else 3
                if self._blocked(curr, direction):
                    return False
        return True

    @staticmethod
    def _supercover_line(start: Cell, end: Cell) -> list[Cell]:
        x0, y0 = start
        x1, y1 = end
        dx = x1 - x0
        dy = y1 - y0
        steps = max(abs(dx), abs(dy))
        if steps == 0:
            return [start]
        cells = []
        for i in range(steps + 1):
            x = int(round(x0 + dx * i / steps))
            y = int(round(y0 + dy * i / steps))
            if not cells or cells[-1] != (x, y):
                cells.append((x, y))
        return cells

    def _blocked(self, pos: Cell, direction: int) -> bool:
        nxt = self._move(pos, direction)
        if not self._in_bounds(pos) or not self._in_bounds(nxt):
            return True
        edge = self._edge(pos, direction)
        return edge in self.state.blocked_edges

    @staticmethod
    def _edge(pos: Cell, direction: int) -> Edge | None:
        x, y = pos
        dx, dy = DIRS[direction]
        nxt = (x + dx, y + dy)
        if not (0 <= nxt[0] < GRID_SIZE and 0 <= nxt[1] < GRID_SIZE):
            return None
        return tuple(sorted((pos, nxt)))  # type: ignore[return-value]

    @staticmethod
    def _move(pos: Cell, direction: int) -> Cell:
        dx, dy = DIRS[direction]
        return pos[0] + dx, pos[1] + dy

    @staticmethod
    def _in_bounds(cell: Cell) -> bool:
        return 0 <= cell[0] < GRID_SIZE and 0 <= cell[1] < GRID_SIZE

    @staticmethod
    def _cell_danger(cell: Cell, hazards: dict[int, set[Cell]], time: int) -> bool:
        return cell in hazards.get(time, set())

    @staticmethod
    def _manhattan(a: Cell, b: Cell) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    @staticmethod
    def _cell(value: object) -> Cell:
        arr = list(value) if value is not None else [0, 0]
        return int(arr[0]), int(arr[1])

    @staticmethod
    def _scalar(value: object, default: float) -> float:
        if value is None:
            return default
        if isinstance(value, (list, tuple, np.ndarray)):
            return float(value[0]) if len(value) else default
        return float(value)
