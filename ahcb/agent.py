from dataclasses import dataclass
from typing import Dict, List, Tuple

from .autoencoder import OnlineSparseAutoencoder
from .cube import CognitiveCube
from .environment import DynamicSandboxWorld, Observation
from .math_utils import argmax
from .reservoir import EchoStateReservoir
from .solver import CubeSolver


@dataclass
class DemoResult:
    steps: int
    total_reward: float
    goals: int
    bumps: int
    hazards: int
    avg_reconstruction_loss: float
    cube_stats: Dict[str, float | int]
    solver_moves: Dict[str, int]
    final_world: str


class AHCB0Agent:
    """Autonomous AHCB-0 learner.

    It has no LLM and no external judge. Its policy is learned from local
    reward readouts over sparse reservoir features.
    """

    def __init__(self, world: DynamicSandboxWorld, seed: int = 31):
        self.world = world
        self.reservoir = EchoStateReservoir(input_size=world.observation_size, seed=seed + 1)
        self.encoder = OnlineSparseAutoencoder(input_size=self.reservoir.size, seed=seed + 2)
        self.cube = CognitiveCube(feature_size=self.encoder.code_size)
        self.solver = CubeSolver(seed=seed + 3)
        self.action_values: Dict[Tuple[int, int], float] = {}
        self.last_features: List[int] = []
        self.last_action = 8
        self.losses: List[float] = []
        self.goals = 0
        self.bumps = 0
        self.hazards = 0

    def act(self, obs: Observation) -> int:
        # A tiny mixture of learned sparse-feature values and hard spatial bias.
        goal_dx = float(obs.tags.get("goal_dx", 0.0))
        goal_dy = float(obs.tags.get("goal_dy", 0.0))
        greedy = self._direction_to_goal(goal_dx, goal_dy)

        scores = []
        active = self.last_features[:]
        for action in range(len(self.world.ACTIONS)):
            learned = sum(self.action_values.get((f, action), 0.0) for f in active)
            bias = 0.25 if action == greedy else 0.0
            if action < 8:
                ray_offset = 4 + action * 4
                wall = obs.vector[ray_offset]
                obstacle = obs.vector[ray_offset + 1]
                hazard = obs.vector[ray_offset + 2]
                goal = obs.vector[ray_offset + 3]
                # Argus-style local sensing: immediate omnidirectional safety
                # has priority over the slower learned value table.
                bias += 0.85 * goal
                bias -= 0.65 * wall
                bias -= 0.75 * obstacle
                bias -= 1.15 * hazard
            if action == 8:
                bias -= 0.05
            scores.append(learned + bias)
        return argmax(scores)

    def learn_from_reward(self, reward: float) -> None:
        for f in self.last_features:
            key = (f, self.last_action)
            old = self.action_values.get(key, 0.0)
            self.action_values[key] = 0.92 * old + 0.08 * reward

    def perceive(self, obs: Observation, reward: float, event: str) -> None:
        reservoir_state = self.reservoir.step(obs.vector)
        sparse = self.encoder.train_step(reservoir_state)
        self.losses.append(sparse.loss)
        self.last_features = sparse.active
        self.cube.ingest(sparse.code, obs.tags, reward, sparse.loss, event)
        self.cube.age_cells()

        # Autonomous internal work: occasionally think without user input.
        if self.world.tick % 5 == 0:
            self.solver.solve_step(self.cube)
        if self.world.tick % 17 == 0:
            self.cube.consolidate_center()
        if self.world.tick % 23 == 0:
            self.cube.prune()

    def run(self, steps: int = 240) -> DemoResult:
        obs = self.world.reset()
        total_reward = 0.0
        self.perceive(obs, 0.0, "reset")

        for _ in range(steps):
            action = self.act(obs)
            self.last_action = action
            obs, reward, _done, info = self.world.step(action)
            event = info.get("event", "move")
            total_reward += reward
            if event == "goal":
                self.goals += 1
            elif event == "blocked":
                self.bumps += 1
            elif event == "hazard":
                self.hazards += 1
            self.learn_from_reward(reward)
            self.perceive(obs, reward, event)

        move_counts: Dict[str, int] = {}
        for move in self.solver.history:
            move_counts[move.name] = move_counts.get(move.name, 0) + 1

        avg_loss = sum(self.losses[-80:]) / max(1, min(80, len(self.losses)))
        return DemoResult(
            steps=steps,
            total_reward=round(total_reward, 3),
            goals=self.goals,
            bumps=self.bumps,
            hazards=self.hazards,
            avg_reconstruction_loss=round(avg_loss, 5),
            cube_stats=self.cube.stats(),
            solver_moves=move_counts,
            final_world=self.world.render(),
        )

    def _direction_to_goal(self, dx: float, dy: float) -> int:
        sx = 0 if abs(dx) < 0.5 else (1 if dx > 0 else -1)
        sy = 0 if abs(dy) < 0.5 else (1 if dy > 0 else -1)
        direction = (sx, sy)
        for i, delta in enumerate(self.world.ACTIONS):
            if delta == direction:
                return i
        return 8
