import json
import random
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
    development_stage: str = "infant"
    exploration_rate: float = 0.0


class AHCB0Agent:
    """Autonomous AHCB-0 learner.

    It has no LLM and no external judge. Its policy is learned from local
    reward readouts over sparse reservoir features.
    """

    def __init__(self, world: DynamicSandboxWorld, seed: int = 31):
        self.seed = seed
        self.rng = random.Random(seed)
        self.world = world
        self.reservoir = EchoStateReservoir(input_size=world.observation_size, seed=seed + 1)
        self.encoder = OnlineSparseAutoencoder(input_size=self.reservoir.size, seed=seed + 2)
        self.cube = CognitiveCube(feature_size=self.encoder.code_size)
        self.solver = CubeSolver(seed=seed + 3)
        self.action_values: Dict[Tuple[int, int], float] = {}
        self.policy_weights: Dict[str, float] = {
            "greedy": 0.25,
            "goal": 0.85,
            "wall": 0.65,
            "obstacle": 0.75,
            "hazard": 1.15,
            "stay": 0.05,
            "curiosity": 0.10,
        }
        self.exploration_rate = 0.18
        self.curiosity = 0.0
        self.total_steps = 0
        self.sleep_cycles = 0
        self.reward_window: List[float] = []
        self.last_features: List[int] = []
        self.last_action = 8
        self.losses: List[float] = []
        self.goals = 0
        self.bumps = 0
        self.hazards = 0

    def act(self, obs: Observation, allow_explore: bool = True) -> int:
        # A tiny mixture of learned sparse-feature values and hard spatial bias.
        scores = self._score_actions(obs)
        if allow_explore and self.rng.random() < self.exploration_rate:
            safe = self._safe_actions(obs)
            return self.rng.choice(safe) if safe else self.rng.randrange(len(self.world.ACTIONS))
        return argmax(scores)

    def _score_actions(self, obs: Observation) -> List[float]:
        goal_dx = float(obs.tags.get("goal_dx", 0.0))
        goal_dy = float(obs.tags.get("goal_dy", 0.0))
        greedy = self._direction_to_goal(goal_dx, goal_dy)

        scores = []
        active = self.last_features[:]
        for action in range(len(self.world.ACTIONS)):
            learned = sum(self.action_values.get((f, action), 0.0) for f in active)
            bias = self.policy_weights["greedy"] if action == greedy else 0.0
            if action < 8:
                ray_offset = 4 + action * 4
                wall = obs.vector[ray_offset]
                obstacle = obs.vector[ray_offset + 1]
                hazard = obs.vector[ray_offset + 2]
                goal = obs.vector[ray_offset + 3]
                # Argus-style local sensing: immediate omnidirectional safety
                # has priority over the slower learned value table.
                bias += self.policy_weights["goal"] * goal
                bias -= self.policy_weights["wall"] * wall
                bias -= self.policy_weights["obstacle"] * obstacle
                bias -= self.policy_weights["hazard"] * hazard
                bias += self.policy_weights["curiosity"] * max(0.0, 0.25 - goal)
            if action == 8:
                bias -= self.policy_weights["stay"]
            scores.append(learned + bias)
        return scores

    def _safe_actions(self, obs: Observation) -> List[int]:
        safe = []
        for action in range(8):
            ray_offset = 4 + action * 4
            wall = obs.vector[ray_offset]
            obstacle = obs.vector[ray_offset + 1]
            hazard = obs.vector[ray_offset + 2]
            if wall < 0.7 and obstacle < 0.7 and hazard < 0.7:
                safe.append(action)
        return safe or [8]

    def learn_from_reward(self, reward: float) -> None:
        for f in self.last_features:
            key = (f, self.last_action)
            old = self.action_values.get(key, 0.0)
            self.action_values[key] = 0.92 * old + 0.08 * reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 200:
            self.reward_window = self.reward_window[-200:]

    def perceive(self, obs: Observation, reward: float, event: str) -> None:
        reservoir_state = self.reservoir.step(obs.vector)
        sparse = self.encoder.train_step(reservoir_state)
        self.losses.append(sparse.loss)
        self.last_features = sparse.active
        self.cube.ingest(sparse.code, obs.tags, reward, sparse.loss, event)
        self.cube.age_cells()
        self.total_steps += 1
        self.curiosity = 0.96 * self.curiosity + 0.04 * sparse.loss

        # Autonomous internal work: occasionally think without user input.
        if self.world.tick % 5 == 0:
            self.solver.solve_step(self.cube)
        if self.world.tick % 17 == 0:
            self.cube.consolidate_center()
        if self.world.tick % 23 == 0:
            self.cube.prune()
        if self.world.tick % 101 == 0:
            self.sleep_cycle()
        if self.world.tick % 53 == 0:
            self.evolve_policy()

    def sleep_cycle(self) -> None:
        """Bounded offline cognition: consolidate and prune without acting."""
        for _ in range(4):
            self.solver.solve_step(self.cube)
        self.cube.consolidate_center()
        self.cube.prune(max_age=20, min_score=0.06)
        self.sleep_cycles += 1

    def evolve_policy(self) -> None:
        """Small self-tuning step, not arbitrary code self-modification."""
        if not self.reward_window:
            return
        recent = self.reward_window[-80:]
        avg = sum(recent) / len(recent)
        blocked = sum(1 for value in recent if value < -0.9)
        jackpot = sum(1 for value in recent if value > 5.0)

        if blocked > max(2, len(recent) // 8):
            self.policy_weights["wall"] = min(2.0, self.policy_weights["wall"] * 1.04)
            self.policy_weights["obstacle"] = min(2.2, self.policy_weights["obstacle"] * 1.05)
            self.policy_weights["hazard"] = min(2.5, self.policy_weights["hazard"] * 1.03)
        if jackpot:
            self.policy_weights["goal"] = min(1.8, self.policy_weights["goal"] * 1.02)
            self.policy_weights["greedy"] = min(0.8, self.policy_weights["greedy"] * 1.01)
        if avg < 0.0:
            self.exploration_rate = min(0.35, self.exploration_rate + 0.015)
        else:
            self.exploration_rate = max(0.03, self.exploration_rate * 0.985)
        self.policy_weights["curiosity"] = max(0.02, min(0.25, 0.08 + self.curiosity))

    def run(self, steps: int = 240) -> DemoResult:
        obs = self.world.reset()
        total_reward = 0.0
        self.perceive(obs, 0.0, "reset")
        run_goals = 0
        run_bumps = 0
        run_hazards = 0

        for _ in range(steps):
            action = self.act(obs)
            self.last_action = action
            obs, reward, _done, info = self.world.step(action)
            event = info.get("event", "move")
            total_reward += reward
            if event == "goal":
                self.goals += 1
                run_goals += 1
            elif event == "blocked":
                self.bumps += 1
                run_bumps += 1
            elif event == "hazard":
                self.hazards += 1
                run_hazards += 1
            self.learn_from_reward(reward)
            self.perceive(obs, reward, event)

        move_counts: Dict[str, int] = {}
        for move in self.solver.history:
            move_counts[move.name] = move_counts.get(move.name, 0) + 1

        avg_loss = sum(self.losses[-80:]) / max(1, min(80, len(self.losses)))
        return DemoResult(
            steps=steps,
            total_reward=round(total_reward, 3),
            goals=run_goals,
            bumps=run_bumps,
            hazards=run_hazards,
            avg_reconstruction_loss=round(avg_loss, 5),
            cube_stats=self.cube.stats(),
            solver_moves=move_counts,
            final_world=self.world.render(),
            development_stage=self.development_stage(),
            exploration_rate=round(self.exploration_rate, 4),
        )

    def _direction_to_goal(self, dx: float, dy: float) -> int:
        sx = 0 if abs(dx) < 0.5 else (1 if dx > 0 else -1)
        sy = 0 if abs(dy) < 0.5 else (1 if dy > 0 else -1)
        direction = (sx, sy)
        for i, delta in enumerate(self.world.ACTIONS):
            if delta == direction:
                return i
        return 8

    def development_stage(self) -> str:
        if self.total_steps < 300:
            return "infant"
        if self.total_steps < 1500:
            return "toddler"
        if self.total_steps < 6000:
            return "child"
        if self.goals < 50:
            return "student"
        return "adult"

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "reservoir": self.reservoir.to_dict(),
            "encoder": self.encoder.to_dict(),
            "cube": self.cube.to_dict(),
            "solver": self.solver.to_dict(),
            "action_values": {f"{k[0]}|{k[1]}": v for k, v in self.action_values.items()},
            "policy_weights": self.policy_weights,
            "exploration_rate": self.exploration_rate,
            "curiosity": self.curiosity,
            "total_steps": self.total_steps,
            "sleep_cycles": self.sleep_cycles,
            "reward_window": self.reward_window[-200:],
            "losses": self.losses[-400:],
            "goals": self.goals,
            "bumps": self.bumps,
            "hazards": self.hazards,
        }

    @classmethod
    def from_dict(cls, world: DynamicSandboxWorld, data: dict) -> "AHCB0Agent":
        agent = cls(world=world, seed=int(data.get("seed", 31)))
        agent.reservoir = EchoStateReservoir.from_dict(dict(data.get("reservoir", {})))
        agent.encoder = OnlineSparseAutoencoder.from_dict(dict(data.get("encoder", {})))
        agent.cube = CognitiveCube.from_dict(dict(data.get("cube", {})))
        agent.solver = CubeSolver.from_dict(dict(data.get("solver", {})))
        agent.action_values = {
            (int(key.split("|")[0]), int(key.split("|")[1])): float(value)
            for key, value in dict(data.get("action_values", {})).items()
        }
        agent.policy_weights.update({str(k): float(v) for k, v in dict(data.get("policy_weights", {})).items()})
        agent.exploration_rate = float(data.get("exploration_rate", agent.exploration_rate))
        agent.curiosity = float(data.get("curiosity", 0.0))
        agent.total_steps = int(data.get("total_steps", 0))
        agent.sleep_cycles = int(data.get("sleep_cycles", 0))
        agent.reward_window = [float(x) for x in data.get("reward_window", [])]
        agent.losses = [float(x) for x in data.get("losses", [])]
        agent.goals = int(data.get("goals", 0))
        agent.bumps = int(data.get("bumps", 0))
        agent.hazards = int(data.get("hazards", 0))
        return agent

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, world: DynamicSandboxWorld, path: str) -> "AHCB0Agent":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(world, data)
