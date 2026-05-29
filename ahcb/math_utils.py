import math
import random
from typing import Iterable, List


def dot(a: Iterable[float], b: Iterable[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def l2_norm(v: Iterable[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def cosine(a: List[float], b: List[float]) -> float:
    denom = l2_norm(a) * l2_norm(b)
    if denom <= 1e-12:
        return 0.0
    return dot(a, b) / denom


def mse(a: List[float], b: List[float]) -> float:
    if not a:
        return 0.0
    return sum((x - y) ** 2 for x, y in zip(a, b)) / len(a)


def argmax(values: List[float]) -> int:
    best_i = 0
    best_v = values[0]
    for i, v in enumerate(values[1:], 1):
        if v > best_v:
            best_i = i
            best_v = v
    return best_i


def top_k_abs(values: List[float], k: int) -> List[int]:
    return sorted(range(len(values)), key=lambda i: abs(values[i]), reverse=True)[:k]


def zeros(n: int) -> List[float]:
    return [0.0 for _ in range(n)]


def random_vector(rng: random.Random, n: int, scale: float = 1.0) -> List[float]:
    return [(rng.random() * 2.0 - 1.0) * scale for _ in range(n)]


def normalize_in_place(v: List[float], target: float = 1.0) -> None:
    norm = l2_norm(v)
    if norm <= 1e-12:
        return
    scale = target / norm
    for i in range(len(v)):
        v[i] *= scale


def clamp(x: float, low: float, high: float) -> float:
    return max(low, min(high, x))

