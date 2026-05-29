import argparse
import random
from dataclasses import dataclass
from typing import Dict, List

from ahcb import AHCB0Agent
from ahcb.curriculum import CURRICULUM, make_world
from ahcb.environment import DynamicSandboxWorld, Observation


@dataclass
class Score:
    reward: float = 0.0
    goals: int = 0
    bumps: int = 0
    hazards: int = 0

    def add(self, reward: float, event: str) -> None:
        self.reward += reward
        if event == "goal":
            self.goals += 1
        elif event == "blocked":
            self.bumps += 1
        elif event == "hazard":
            self.hazards += 1


class RandomController:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)

    def act(self, world: DynamicSandboxWorld, _obs: Observation) -> int:
        return self.rng.randrange(len(world.ACTIONS))


class ReflexController:
    """Argus-like reflex baseline: use sensors, no memory."""

    def act(self, world: DynamicSandboxWorld, obs: Observation) -> int:
        goal_dx = float(obs.tags.get("goal_dx", 0.0))
        goal_dy = float(obs.tags.get("goal_dy", 0.0))
        sx = 0 if abs(goal_dx) < 0.5 else (1 if goal_dx > 0 else -1)
        sy = 0 if abs(goal_dy) < 0.5 else (1 if goal_dy > 0 else -1)
        greedy = world.ACTIONS.index((sx, sy)) if (sx, sy) in world.ACTIONS else 8

        best_action = 8
        best_score = -999.0
        for action in range(len(world.ACTIONS)):
            score = 0.35 if action == greedy else 0.0
            if action < 8:
                offset = 4 + action * 4
                wall = obs.vector[offset]
                obstacle = obs.vector[offset + 1]
                hazard = obs.vector[offset + 2]
                goal = obs.vector[offset + 3]
                score += 1.0 * goal - 0.8 * wall - 0.9 * obstacle - 1.2 * hazard
            else:
                score -= 0.05
            if score > best_score:
                best_score = score
                best_action = action
        return best_action


def rollout_controller(world: DynamicSandboxWorld, controller, steps: int) -> Score:
    obs = world.reset()
    score = Score()
    for _ in range(steps):
        action = controller.act(world, obs)
        obs, reward, _done, info = world.step(action)
        score.add(reward, info.get("event", "move"))
    return score


def train_ahcb(level: int, seed: int, train_steps: int) -> AHCB0Agent:
    world = make_world(level, seed)
    agent = AHCB0Agent(world, seed=seed + 10_000)
    if train_steps > 0:
        agent.run(train_steps)
    agent.exploration_rate = 0.0
    return agent


def eval_ahcb(agent: AHCB0Agent, level: int, seed: int, steps: int) -> Score:
    agent.world = make_world(level, seed)
    obs = agent.world.reset()
    score = Score()
    agent.perceive(obs, 0.0, "eval_reset")
    old_explore = agent.exploration_rate
    agent.exploration_rate = 0.0
    try:
        for _ in range(steps):
            action = agent.act(obs, allow_explore=False)
            agent.last_action = action
            obs, reward, _done, info = agent.world.step(action)
            event = info.get("event", "move")
            score.add(reward, event)
            # Keep learning while interacting, but deterministic action choice.
            agent.learn_from_reward(reward)
            agent.perceive(obs, reward, event)
    finally:
        agent.exploration_rate = old_explore
    return score


def mean(scores: List[Score]) -> Score:
    n = max(1, len(scores))
    return Score(
        reward=sum(s.reward for s in scores) / n,
        goals=round(sum(s.goals for s in scores) / n),
        bumps=round(sum(s.bumps for s in scores) / n),
        hazards=round(sum(s.hazards for s in scores) / n),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark AHCB-0 against simple baselines.")
    parser.add_argument("--levels", type=int, default=4, help="Number of curriculum levels to test.")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=220)
    parser.add_argument("--train-steps", type=int, default=700)
    args = parser.parse_args()

    print("AHCB-0 benchmark")
    print("=" * 72)
    print(f"eval steps={args.steps} seeds={args.seeds} ahcb pretrain={args.train_steps}")
    print()
    print(f"{'level':14s} {'agent':10s} {'reward':>9s} {'goals':>6s} {'bumps':>6s} {'hazards':>7s}")
    print("-" * 72)

    for level in range(min(args.levels, len(CURRICULUM))):
        spec = CURRICULUM[level]
        rows: Dict[str, List[Score]] = {"random": [], "reflex": [], "ahcb": []}
        for seed in range(args.seeds):
            eval_seed = 1000 + level * 100 + seed
            rows["random"].append(
                rollout_controller(make_world(level, eval_seed), RandomController(eval_seed), args.steps)
            )
            rows["reflex"].append(
                rollout_controller(make_world(level, eval_seed), ReflexController(), args.steps)
            )
            agent = train_ahcb(level, 5000 + level * 100 + seed, args.train_steps)
            rows["ahcb"].append(eval_ahcb(agent, level, eval_seed, args.steps))

        for name in ("random", "reflex", "ahcb"):
            s = mean(rows[name])
            print(
                f"{spec.name:14s} {name:10s} {s.reward:9.2f} "
                f"{s.goals:6d} {s.bumps:6d} {s.hazards:7d}"
            )
        print("-" * 72)


if __name__ == "__main__":
    main()

