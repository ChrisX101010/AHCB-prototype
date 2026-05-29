from ahcb import AHCB0Agent, DynamicSandboxWorld


def main() -> None:
    world = DynamicSandboxWorld(seed=42)
    agent = AHCB0Agent(world, seed=100)
    result = agent.run(steps=260)

    print("AHCB-0 autonomous cognitive-cube demo")
    print("=" * 44)
    print(f"steps                  : {result.steps}")
    print(f"total_reward           : {result.total_reward}")
    print(f"goals                  : {result.goals}")
    print(f"blocked bumps          : {result.bumps}")
    print(f"hazard hits            : {result.hazards}")
    print(f"avg reconstruction loss: {result.avg_reconstruction_loss}")
    print()
    print("cube stats")
    for key, value in result.cube_stats.items():
        print(f"  {key:16s}: {value}")
    print()
    print("solver moves")
    for key, value in sorted(result.solver_moves.items()):
        print(f"  {key:16s}: {value}")
    print()
    print("final sandbox")
    print(result.final_world)


if __name__ == "__main__":
    main()

