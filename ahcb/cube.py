from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .math_utils import cosine, zeros


Address = Tuple[int, int, int]


@dataclass
class CubeCell:
    vector: List[float]
    salience: float = 0.0
    confidence: float = 0.0
    utility: float = 0.0
    visits: int = 0
    age: int = 0
    last_seen: int = 0
    labels: Dict[str, int] = field(default_factory=dict)

    def update(self, code: List[float], reward: float, loss: float, label: str, tick: int) -> None:
        self.visits += 1
        self.age = 0
        self.last_seen = tick
        alpha = 1.0 / min(12, self.visits)
        for i, v in enumerate(code):
            self.vector[i] = (1.0 - alpha) * self.vector[i] + alpha * v
        self.salience = 0.92 * self.salience + 0.08 * (abs(reward) + loss)
        self.utility = 0.94 * self.utility + 0.06 * reward
        self.confidence = min(1.0, self.confidence + 0.04)
        self.labels[label] = self.labels.get(label, 0) + 1


class CognitiveCube:
    """3D memory: region x modality x abstraction level."""

    def __init__(self, regions: int = 6, modalities: int = 3, levels: int = 4, feature_size: int = 32):
        self.regions = regions
        self.modalities = modalities
        self.levels = levels
        self.feature_size = feature_size
        self.cells: Dict[Address, CubeCell] = {}
        self.links: Dict[Tuple[Address, Address], float] = {}
        self.center = zeros(feature_size)
        self.tick = 0
        self.last_address: Optional[Address] = None
        self.pruned_cells = 0
        self.pruned_links = 0
        self.consolidations = 0

    def route(self, code: List[float], tags: Dict[str, float | str | int], reward: float, loss: float) -> Address:
        active = max(range(len(code)), key=lambda i: abs(code[i])) if code else 0
        goal_signal = float(tags.get("goal_signal", 0.0))
        hazard_signal = float(tags.get("hazard_signal", 0.0))
        open_ratio = float(tags.get("open_ratio", 0.0))

        if reward > 2.0 or goal_signal > 0.3:
            region = 0  # reward / goal cortex
        elif reward < -0.8 or hazard_signal > 0.3:
            region = 1  # danger cortex
        elif open_ratio < 0.35:
            region = 2  # obstacle / pressure cortex
        else:
            region = 3 + (active % max(1, self.regions - 3))

        modality = 0
        if abs(float(tags.get("goal_dx", 0))) + abs(float(tags.get("goal_dy", 0))) < 4:
            modality = 1
        if loss > 0.09:
            modality = 2

        if loss > 0.16:
            level = 0      # raw surprise
        elif loss > 0.08:
            level = 1      # pattern
        elif abs(reward) > 0.5:
            level = 2      # consequence
        else:
            level = 3      # stable abstraction
        return (region % self.regions, modality % self.modalities, level % self.levels)

    def ingest(self, code: List[float], tags: Dict[str, float | str | int], reward: float, loss: float, label: str) -> Address:
        self.tick += 1
        addr = self.route(code, tags, reward, loss)
        cell = self.cells.get(addr)
        if cell is None:
            cell = CubeCell(vector=zeros(self.feature_size))
            self.cells[addr] = cell
        cell.update(code, reward, loss, label, self.tick)

        if self.last_address is not None and self.last_address != addr:
            key = (self.last_address, addr)
            self.links[key] = self.links.get(key, 0.0) * 0.98 + 1.0
        self.last_address = addr
        return addr

    def age_cells(self) -> None:
        for cell in self.cells.values():
            cell.age += 1
            cell.salience *= 0.995
            cell.confidence *= 0.999
        for key in list(self.links):
            self.links[key] *= 0.997

    def consolidate_center(self) -> float:
        if not self.cells:
            return 0.0
        total_w = 0.0
        new_center = zeros(self.feature_size)
        for cell in self.cells.values():
            w = max(0.02, cell.salience + cell.confidence + max(0.0, cell.utility))
            total_w += w
            for i, v in enumerate(cell.vector):
                new_center[i] += w * v
        if total_w > 0:
            for i in range(self.feature_size):
                new_center[i] /= total_w
        shift = 1.0 - cosine(self.center, new_center)
        self.center = new_center
        self.consolidations += 1
        return shift

    def rotate_region(self, region: int) -> int:
        """A Rubik-like move: rotate a region's modality/level addresses."""
        moves: List[Tuple[Address, Address]] = []
        for m in range(self.modalities):
            for l in range(self.levels):
                old = (region, m, l)
                new = (region, (m + 1) % self.modalities, l)
                if old in self.cells:
                    moves.append((old, new))
        if not moves:
            return 0
        old_cells = dict(self.cells)
        for old, new in moves:
            self.cells[new] = old_cells[old]
            if old != new and old in self.cells:
                del self.cells[old]
        self._rewrite_links(dict(moves))
        return len(moves)

    def lift_surprise(self) -> int:
        """Move high-surprise raw cells toward pattern level after consolidation."""
        moves = []
        for addr, cell in list(self.cells.items()):
            r, m, l = addr
            if l == 0 and cell.visits >= 2:
                moves.append((addr, (r, m, 1)))
        for old, new in moves:
            if new in self.cells:
                self._merge_cells(new, old)
            else:
                self.cells[new] = self.cells.pop(old)
        self._rewrite_links(dict(moves))
        return len(moves)

    def prune(self, max_age: int = 35, min_score: float = 0.04) -> int:
        removed = 0
        for addr, cell in list(self.cells.items()):
            score = cell.salience + abs(cell.utility) + 0.2 * cell.confidence
            if cell.age > max_age and score < min_score:
                del self.cells[addr]
                removed += 1
        if removed:
            live = set(self.cells)
            for key, weight in list(self.links.items()):
                if key[0] not in live or key[1] not in live or weight < 0.02:
                    del self.links[key]
                    self.pruned_links += 1
        self.pruned_cells += removed
        return removed

    def _merge_cells(self, keep: Address, drop: Address) -> None:
        a = self.cells[keep]
        b = self.cells.pop(drop)
        total = a.visits + b.visits
        if total <= 0:
            return
        for i in range(self.feature_size):
            a.vector[i] = (a.vector[i] * a.visits + b.vector[i] * b.visits) / total
        a.salience = max(a.salience, b.salience)
        a.utility = (a.utility + b.utility) / 2.0
        a.confidence = max(a.confidence, b.confidence)
        a.visits = total

    def _rewrite_links(self, mapping: Dict[Address, Address]) -> None:
        if not mapping:
            return
        new_links: Dict[Tuple[Address, Address], float] = {}
        for (a, b), w in self.links.items():
            na = mapping.get(a, a)
            nb = mapping.get(b, b)
            if na in self.cells and nb in self.cells and na != nb:
                new_links[(na, nb)] = new_links.get((na, nb), 0.0) + w
        self.links = new_links

    def stats(self) -> Dict[str, float | int]:
        return {
            "cells": len(self.cells),
            "links": len(self.links),
            "pruned_cells": self.pruned_cells,
            "pruned_links": self.pruned_links,
            "consolidations": self.consolidations,
            "center_energy": sum(abs(x) for x in self.center),
        }

