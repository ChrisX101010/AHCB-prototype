import os
import tempfile

from ahcb import AHCB0Agent, DynamicSandboxWorld
from ahcb.curriculum import make_world


def assert_true(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        raise AssertionError(f"{name} failed. {detail}")
    print(f"[ok] {name}")


def test_world_observation_size() -> None:
    world = DynamicSandboxWorld(seed=1)
    obs = world.reset()
    assert_true("observation size", len(obs.vector) == world.observation_size)


def test_autonomous_learning_loop() -> None:
    world = DynamicSandboxWorld(seed=2)
    agent = AHCB0Agent(world, seed=3)
    result = agent.run(steps=140)
    assert_true("cube formed cells", result.cube_stats["cells"] > 0, str(result.cube_stats))
    assert_true("solver ran", sum(result.solver_moves.values()) > 0, str(result.solver_moves))
    assert_true("autoencoder loss finite", 0.0 <= result.avg_reconstruction_loss < 10.0, str(result.avg_reconstruction_loss))


def test_pruning_and_consolidation() -> None:
    world = DynamicSandboxWorld(seed=4)
    agent = AHCB0Agent(world, seed=5)
    result = agent.run(steps=260)
    stats = result.cube_stats
    assert_true("center consolidated", stats["consolidations"] > 0, str(stats))
    assert_true("memory bounded", stats["cells"] <= 72, str(stats))


def test_save_load_roundtrip() -> None:
    world = make_world(1, seed=6)
    agent = AHCB0Agent(world, seed=7)
    agent.run(steps=80)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "brain.json")
        agent.save(path)
        loaded = AHCB0Agent.load(make_world(1, seed=6), path)
        assert_true("loaded steps", loaded.total_steps == agent.total_steps)
        assert_true("loaded cube cells", len(loaded.cube.cells) == len(agent.cube.cells))
        assert_true("loaded policy", loaded.policy_weights["goal"] > 0)


if __name__ == "__main__":
    test_world_observation_size()
    test_autonomous_learning_loop()
    test_pruning_and_consolidation()
    test_save_load_roundtrip()
    print("all tests passed")
