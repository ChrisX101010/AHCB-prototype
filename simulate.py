import argparse
import os
import time

from ahcb import AHCB0Agent, DynamicSandboxWorld


def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch AHCB-0 run in the toy sandbox.")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--live", action="store_true", help="Clear the terminal between frames.")
    args = parser.parse_args()

    world = DynamicSandboxWorld(seed=args.seed)
    agent = AHCB0Agent(world, seed=args.seed + 100)
    obs = world.reset()
    agent.perceive(obs, 0.0, "reset")
    total_reward = 0.0

    for step in range(1, args.steps + 1):
        action = agent.act(obs)
        agent.last_action = action
        obs, reward, _done, info = world.step(action)
        event = info.get("event", "move")
        total_reward += reward
        if event == "goal":
            agent.goals += 1
        elif event == "blocked":
            agent.bumps += 1
        elif event == "hazard":
            agent.hazards += 1
        agent.learn_from_reward(reward)
        agent.perceive(obs, reward, event)

        if args.live:
            clear_screen()
        stats = agent.cube.stats()
        action_name = world.ACTION_NAMES[action]
        print(f"step {step:03d}  action={action_name:<4s} event={event:<7s} reward={reward:+.2f}")
        print(
            f"total={total_reward:+.2f} goals={agent.goals} bumps={agent.bumps} "
            f"hazards={agent.hazards} loss={agent.losses[-1]:.5f}"
        )
        print(
            f"cube cells={stats['cells']} links={stats['links']} "
            f"pruned={stats['pruned_cells']} center_energy={stats['center_energy']:.3f}"
        )
        if agent.solver.history:
            move = agent.solver.history[-1]
            print(
                f"last solver move={move.name} "
                f"pressure {move.score_before:.2f}->{move.score_after:.2f}"
            )
        print(world.render())
        print("-" * 44)
        if args.delay > 0:
            time.sleep(args.delay)


if __name__ == "__main__":
    main()

