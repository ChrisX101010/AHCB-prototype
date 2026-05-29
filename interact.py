from ahcb import AHCB0Agent
from ahcb.curriculum import CURRICULUM, make_world


def main() -> None:
    level = 1
    world = make_world(level, seed=77)
    agent = AHCB0Agent(world, seed=177)
    obs = world.reset()
    agent.perceive(obs, 0.0, "interactive_reset")

    print("AHCB-0 interactive sandbox")
    print("commands: step, run N, look, stats, sleep, level N, save PATH, load PATH, quit")
    print(world.render())

    while True:
        command = input("ahcb> ").strip()
        if not command:
            continue
        if command in {"quit", "exit", "q"}:
            break
        if command == "look":
            print(world.render())
            continue
        if command == "stats":
            print({
                "stage": agent.development_stage(),
                "steps": agent.total_steps,
                "goals": agent.goals,
                "bumps": agent.bumps,
                "hazards": agent.hazards,
                "exploration": round(agent.exploration_rate, 4),
                "cube": agent.cube.stats(),
                "policy": {k: round(v, 3) for k, v in agent.policy_weights.items()},
            })
            continue
        if command == "sleep":
            agent.sleep_cycle()
            print("sleep cycle complete", agent.cube.stats())
            continue
        if command.startswith("level "):
            level = int(command.split(maxsplit=1)[1])
            world = make_world(level, seed=77 + agent.total_steps)
            agent.world = world
            obs = world.reset()
            agent.perceive(obs, 0.0, "level_reset")
            print(f"level set to {CURRICULUM[level].name}")
            print(world.render())
            continue
        if command.startswith("save "):
            path = command.split(maxsplit=1)[1]
            agent.save(path)
            print(f"saved {path}")
            continue
        if command.startswith("load "):
            path = command.split(maxsplit=1)[1]
            agent = AHCB0Agent.load(world, path)
            print(f"loaded {path}")
            continue
        if command.startswith("run "):
            count = int(command.split(maxsplit=1)[1])
        elif command == "step":
            count = 1
        else:
            print("unknown command")
            continue

        for _ in range(count):
            action = agent.act(obs)
            agent.last_action = action
            obs, reward, _done, info = world.step(action)
            event = info.get("event", "move")
            agent.learn_from_reward(reward)
            agent.perceive(obs, reward, event)
            print(f"{world.ACTION_NAMES[action]:<4s} {event:<7s} reward={reward:+.2f}")
        print(world.render())


if __name__ == "__main__":
    main()

