import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .cube import CognitiveCube


@dataclass
class SolverMove:
    name: str
    score_before: float
    score_after: float
    changed: int


class CubeSolver:
    """Rubik/Stockfish-inspired internal state solver.

    It does shallow move search over cognitive-cube transformations and caches
    which moves helped in similar pressure states.
    """

    def __init__(self, seed: int = 23):
        self.rng = random.Random(seed)
        self.cache: Dict[Tuple[int, int, int], str] = {}
        self.history: List[SolverMove] = []

    def pressure(self, cube: CognitiveCube) -> float:
        if not cube.cells:
            return 0.0
        raw = 0.0
        stale = 0.0
        weak = 0.0
        for (_r, _m, level), cell in cube.cells.items():
            if level == 0:
                raw += 1.0
            if cell.age > 15:
                stale += 1.0
            if cell.confidence < 0.08:
                weak += 1.0
        return raw * 1.4 + stale * 0.35 + weak * 0.2 + len(cube.links) * 0.005

    def signature(self, cube: CognitiveCube) -> Tuple[int, int, int]:
        raw = sum(1 for (_r, _m, l) in cube.cells if l == 0)
        stale = sum(1 for c in cube.cells.values() if c.age > 15)
        links = len(cube.links)
        return (min(9, raw), min(9, stale), min(9, links // 3))

    def solve_step(self, cube: CognitiveCube) -> SolverMove:
        before = self.pressure(cube)
        sig = self.signature(cube)
        preferred = self.cache.get(sig)

        candidates = ["lift", "consolidate", "prune"]
        candidates.extend([f"rotate:{r}" for r in range(cube.regions)])
        if preferred in candidates:
            candidates.remove(preferred)
            candidates.insert(0, preferred)

        best = None
        best_delta = -10**9
        for name in candidates[:5]:
            delta_hint = self._estimate_move(cube, name)
            if delta_hint > best_delta:
                best = name
                best_delta = delta_hint

        changed = self._apply(cube, best or "consolidate")
        after = self.pressure(cube)
        move = SolverMove(name=best or "consolidate", score_before=before, score_after=after, changed=changed)
        self.history.append(move)
        if after < before:
            self.cache[sig] = move.name
        return move

    def to_dict(self) -> dict:
        return {
            "cache": {"|".join(map(str, k)): v for k, v in self.cache.items()},
            "history": [
                {
                    "name": m.name,
                    "score_before": m.score_before,
                    "score_after": m.score_after,
                    "changed": m.changed,
                }
                for m in self.history[-500:]
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CubeSolver":
        obj = cls()
        obj.cache = {
            tuple(int(x) for x in key.split("|")): str(value)
            for key, value in dict(data.get("cache", {})).items()
        }
        obj.history = [
            SolverMove(
                name=str(m.get("name", "?")),
                score_before=float(m.get("score_before", 0.0)),
                score_after=float(m.get("score_after", 0.0)),
                changed=int(m.get("changed", 0)),
            )
            for m in data.get("history", [])
        ]
        return obj

    def _estimate_move(self, cube: CognitiveCube, name: str) -> float:
        if name == "lift":
            return sum(1.0 for (_r, _m, l), c in cube.cells.items() if l == 0 and c.visits >= 2)
        if name == "prune":
            return sum(0.5 for c in cube.cells.values() if c.age > 25 and c.salience < 0.04)
        if name == "consolidate":
            return 0.8 if cube.cells else 0.0
        if name.startswith("rotate:"):
            region = int(name.split(":", 1)[1])
            return sum(0.15 for (r, _m, _l) in cube.cells if r == region)
        return 0.0

    def _apply(self, cube: CognitiveCube, name: str) -> int:
        if name == "lift":
            return cube.lift_surprise()
        if name == "prune":
            return cube.prune()
        if name == "consolidate":
            cube.consolidate_center()
            return 1
        if name.startswith("rotate:"):
            return cube.rotate_region(int(name.split(":", 1)[1]))
        return 0
