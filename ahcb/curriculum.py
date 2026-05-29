from dataclasses import dataclass
from typing import List

from .environment import DynamicSandboxWorld


@dataclass(frozen=True)
class CurriculumLevel:
    name: str
    width: int
    height: int
    obstacles: int
    hazards: int
    drift_interval: int
    drift_chance: float


CURRICULUM: List[CurriculumLevel] = [
    CurriculumLevel("infant-room", 7, 7, 4, 1, 0, 0.0),
    CurriculumLevel("toddler-room", 8, 8, 8, 2, 11, 0.15),
    CurriculumLevel("child-yard", 9, 9, 12, 4, 9, 0.25),
    CurriculumLevel("student-maze", 11, 11, 20, 7, 7, 0.35),
    CurriculumLevel("adult-field", 13, 13, 32, 11, 5, 0.45),
]


def make_world(level: int, seed: int) -> DynamicSandboxWorld:
    spec = CURRICULUM[max(0, min(level, len(CURRICULUM) - 1))]
    return DynamicSandboxWorld(
        width=spec.width,
        height=spec.height,
        seed=seed,
        obstacle_count=spec.obstacles,
        hazard_count=spec.hazards,
        drift_interval=spec.drift_interval,
        drift_chance=spec.drift_chance,
    )

