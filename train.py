import argparse
import json
import os
import time
from datetime import UTC, datetime

from ahcb import AHCB0Agent
from ahcb.curriculum import CURRICULUM, make_world


def choose_level(total_steps: int, explicit_level: int | None) -> int:
    if explicit_level is not None:
        return max(0, min(explicit_level, len(CURRICULUM) - 1))
    if total_steps < 2_000:
        return 0
    if total_steps < 7_000:
        return 1
    if total_steps < 20_000:
        return 2
    if total_steps < 60_000:
        return 3
    return 4


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous AHCB-0 training loop.")
    parser.add_argument("--steps", type=int, default=5_000)
    parser.add_argument("--chunk", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--level", type=int, default=None)
    parser.add_argument("--save", default="runs/ahcb_brain.json")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--metrics", default="runs/metrics.jsonl")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.metrics) or ".", exist_ok=True)

    level = choose_level(0, args.level)
    world = make_world(level, args.seed)
    if args.resume:
        agent = AHCB0Agent.load(world, args.resume)
    else:
        agent = AHCB0Agent(world, seed=args.seed + 100)

    started = time.time()
    steps_left = args.steps
    chunk_index = 0
    while steps_left > 0:
        level = choose_level(agent.total_steps, args.level)
        agent.world = make_world(level, args.seed + chunk_index)
        n = min(args.chunk, steps_left)
        result = agent.run(n)
        steps_left -= n
        chunk_index += 1
        agent.save(args.save)

        record = {
            "ts": datetime.now(UTC).isoformat(),
            "chunk": chunk_index,
            "level": CURRICULUM[level].name,
            "steps_this_chunk": n,
            "total_steps": agent.total_steps,
            "stage": agent.development_stage(),
            "reward": result.total_reward,
            "goals": result.goals,
            "bumps": result.bumps,
            "hazards": result.hazards,
            "loss": result.avg_reconstruction_loss,
            "exploration": result.exploration_rate,
            "cube": result.cube_stats,
            "sleep_cycles": agent.sleep_cycles,
        }
        with open(args.metrics, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        elapsed = time.time() - started
        print(
            f"[{chunk_index}] level={record['level']} stage={record['stage']} "
            f"steps={agent.total_steps} reward={result.total_reward:+.2f} "
            f"goals={result.goals} bumps={result.bumps} hazards={result.hazards} "
            f"loss={result.avg_reconstruction_loss:.5f} elapsed={elapsed:.1f}s"
        )

    print(f"saved brain: {args.save}")
    print(f"saved metrics: {args.metrics}")


if __name__ == "__main__":
    main()
