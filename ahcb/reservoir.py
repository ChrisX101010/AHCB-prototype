import math
import random
from typing import List, Tuple

from .math_utils import dot


class EchoStateReservoir:
    """A fixed recurrent reservoir with a trainable downstream readout.

    This gives AHCB-0 cheap temporal dynamics without backpropagating through
    time. The internal recurrent graph is intentionally left random.
    """

    def __init__(
        self,
        input_size: int,
        size: int = 96,
        sparsity: float = 0.08,
        spectral_radius: float = 0.85,
        leak: float = 0.45,
        seed: int = 11,
    ):
        self.input_size = input_size
        self.size = size
        self.leak = leak
        self.rng = random.Random(seed)
        self.state = [0.0 for _ in range(size)]
        self.win = [
            [(self.rng.random() * 2.0 - 1.0) * 0.45 for _ in range(input_size)]
            for _ in range(size)
        ]
        self.recurrent: List[List[Tuple[int, float]]] = []
        fanout = max(1, int(size * sparsity))
        for _ in range(size):
            row = []
            for _ in range(fanout):
                j = self.rng.randrange(size)
                w = self.rng.random() * 2.0 - 1.0
                row.append((j, w))
            norm = sum(abs(w) for _, w in row) or 1.0
            row = [(j, w * spectral_radius / norm) for j, w in row]
            self.recurrent.append(row)

    def reset(self) -> None:
        for i in range(self.size):
            self.state[i] = 0.0

    def step(self, vector: List[float]) -> List[float]:
        nxt = [0.0 for _ in range(self.size)]
        for i in range(self.size):
            recurrent_sum = sum(w * self.state[j] for j, w in self.recurrent[i])
            total = dot(self.win[i], vector) + recurrent_sum
            candidate = math.tanh(total)
            nxt[i] = (1.0 - self.leak) * self.state[i] + self.leak * candidate
        self.state = nxt
        return list(self.state)

    def to_dict(self) -> dict:
        return {
            "input_size": self.input_size,
            "size": self.size,
            "leak": self.leak,
            "state": self.state,
            "win": self.win,
            "recurrent": self.recurrent,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EchoStateReservoir":
        obj = cls(
            input_size=int(data.get("input_size", 39)),
            size=int(data.get("size", 96)),
            leak=float(data.get("leak", 0.45)),
        )
        obj.state = [float(x) for x in data.get("state", obj.state)]
        obj.win = [[float(x) for x in row] for row in data.get("win", obj.win)]
        obj.recurrent = [
            [(int(j), float(w)) for j, w in row]
            for row in data.get("recurrent", obj.recurrent)
        ]
        return obj
