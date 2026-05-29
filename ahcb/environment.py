import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple


Pos = Tuple[int, int]


@dataclass
class Observation:
    vector: List[float]
    tags: Dict[str, float | str | int]


class DynamicSandboxWorld:
    """A small non-static world with Argus-like omnidirectional sensing.

    The agent sees in eight directions at once. Objects drift over time, so the
    world is not just a memorized maze.
    """

    ACTIONS: List[Pos] = [
        (0, -1),   # N
        (1, -1),   # NE
        (1, 0),    # E
        (1, 1),    # SE
        (0, 1),    # S
        (-1, 1),   # SW
        (-1, 0),   # W
        (-1, -1),  # NW
        (0, 0),    # stay
    ]
    ACTION_NAMES = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "STAY"]

    def __init__(self, width: int = 9, height: int = 9, seed: int = 7):
        self.width = width
        self.height = height
        self.rng = random.Random(seed)
        self.tick = 0
        self.last_reward = 0.0
        self.agent: Pos = (1, 1)
        self.goal: Pos = (width - 2, height - 2)
        self.obstacles: Set[Pos] = set()
        self.hazards: Set[Pos] = set()
        self.reset()

    @property
    def observation_size(self) -> int:
        # agent xy, goal bearing xy, 8 rays * 4 object channels, reward,
        # time sin/cos, local open-space ratio
        return 2 + 2 + 8 * 4 + 1 + 2 + 1

    def reset(self) -> Observation:
        self.tick = 0
        self.last_reward = 0.0
        self.agent = (1, 1)
        self.goal = (self.width - 2, self.height - 2)
        self.obstacles = set()
        self.hazards = set()

        while len(self.obstacles) < 12:
            p = self._random_empty()
            if p not in (self.agent, self.goal):
                self.obstacles.add(p)
        while len(self.hazards) < 5:
            p = self._random_empty()
            if p not in (self.agent, self.goal):
                self.hazards.add(p)
        return self.observe()

    def _random_empty(self) -> Pos:
        for _ in range(1000):
            p = (self.rng.randrange(1, self.width - 1),
                 self.rng.randrange(1, self.height - 1))
            if p != self.agent and p != self.goal and p not in self.obstacles and p not in self.hazards:
                return p
        return (1, 1)

    def observe(self) -> Observation:
        ax, ay = self.agent
        gx, gy = self.goal
        max_d = max(self.width, self.height)

        vector: List[float] = [
            ax / (self.width - 1),
            ay / (self.height - 1),
            (gx - ax) / max_d,
            (gy - ay) / max_d,
        ]

        ray_channels: List[float] = []
        best_goal_signal = 0.0
        best_hazard_signal = 0.0
        open_count = 0
        for dx, dy in self.ACTIONS[:8]:
            hit_kind = "empty"
            hit_dist = max_d
            x, y = ax, ay
            for dist in range(1, max_d + 1):
                x += dx
                y += dy
                if not self._inside(x, y):
                    hit_kind = "wall"
                    hit_dist = dist
                    break
                p = (x, y)
                if p == self.goal:
                    hit_kind = "goal"
                    hit_dist = dist
                    break
                if p in self.hazards:
                    hit_kind = "hazard"
                    hit_dist = dist
                    break
                if p in self.obstacles:
                    hit_kind = "obstacle"
                    hit_dist = dist
                    break
            signal = 1.0 / hit_dist
            ray_channels.extend([
                signal if hit_kind == "wall" else 0.0,
                signal if hit_kind == "obstacle" else 0.0,
                signal if hit_kind == "hazard" else 0.0,
                signal if hit_kind == "goal" else 0.0,
            ])
            if hit_kind == "goal":
                best_goal_signal = max(best_goal_signal, signal)
            if hit_kind == "hazard":
                best_hazard_signal = max(best_hazard_signal, signal)
            if hit_kind in ("empty", "goal"):
                open_count += 1

        vector.extend(ray_channels)
        vector.append(self.last_reward / 10.0)
        vector.append(math.sin(self.tick / 6.0))
        vector.append(math.cos(self.tick / 6.0))
        vector.append(open_count / 8.0)

        tags: Dict[str, float | str | int] = {
            "x": ax,
            "y": ay,
            "goal_dx": gx - ax,
            "goal_dy": gy - ay,
            "goal_signal": best_goal_signal,
            "hazard_signal": best_hazard_signal,
            "open_ratio": open_count / 8.0,
            "tick": self.tick,
        }
        return Observation(vector=vector, tags=tags)

    def step(self, action: int) -> Tuple[Observation, float, bool, Dict[str, str]]:
        action = max(0, min(len(self.ACTIONS) - 1, action))
        old_dist = self._goal_distance(self.agent)
        dx, dy = self.ACTIONS[action]
        nx, ny = self.agent[0] + dx, self.agent[1] + dy

        reward = -0.05
        event = "move"
        if not self._inside(nx, ny) or (nx, ny) in self.obstacles:
            reward -= 1.0
            event = "blocked"
        else:
            self.agent = (nx, ny)
            if self.agent in self.hazards:
                reward -= 2.0
                event = "hazard"
            new_dist = self._goal_distance(self.agent)
            if new_dist < old_dist:
                reward += 0.15
            elif new_dist > old_dist:
                reward -= 0.08
            if self.agent == self.goal:
                reward += 10.0
                event = "goal"
                self.goal = self._random_empty()

        self.tick += 1
        self.last_reward = reward
        self._drift_world()
        return self.observe(), reward, False, {"event": event}

    def _inside(self, x: int, y: int) -> bool:
        return 0 < x < self.width - 1 and 0 < y < self.height - 1

    def _goal_distance(self, pos: Pos) -> int:
        return abs(self.goal[0] - pos[0]) + abs(self.goal[1] - pos[1])

    def _drift_world(self) -> None:
        if self.tick % 7 != 0:
            return
        new_obstacles: Set[Pos] = set()
        for p in self.obstacles:
            if self.rng.random() < 0.35:
                dx, dy = self.rng.choice(self.ACTIONS[:8])
                q = (p[0] + dx, p[1] + dy)
                if self._inside(q[0], q[1]) and q not in self.hazards and q != self.agent and q != self.goal:
                    new_obstacles.add(q)
                else:
                    new_obstacles.add(p)
            else:
                new_obstacles.add(p)
        self.obstacles = new_obstacles

        if self.tick % 21 == 0 and self.hazards:
            self.hazards.pop()
            self.hazards.add(self._random_empty())

    def render(self) -> str:
        rows = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                p = (x, y)
                if p == self.agent:
                    row.append("A")
                elif p == self.goal:
                    row.append("G")
                elif p in self.hazards:
                    row.append("!")
                elif p in self.obstacles:
                    row.append("#")
                elif x == 0 or y == 0 or x == self.width - 1 or y == self.height - 1:
                    row.append("+")
                else:
                    row.append(".")
            rows.append("".join(row))
        return "\n".join(rows)
