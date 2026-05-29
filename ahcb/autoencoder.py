import math
import random
from dataclasses import dataclass
from typing import List

from .math_utils import dot, mse, normalize_in_place, top_k_abs


@dataclass
class SparseCode:
    code: List[float]
    reconstruction: List[float]
    loss: float
    active: List[int]


class OnlineSparseAutoencoder:
    """A tiny k-sparse autoencoder-style compressor.

    It is intentionally simple: a random encoder produces a sparse code, and a
    decoder learns online to reconstruct reservoir states. This behaves like a
    local feature compressor rather than a full deep net.
    """

    def __init__(
        self,
        input_size: int,
        code_size: int = 32,
        k: int = 6,
        lr: float = 0.035,
        seed: int = 17,
    ):
        self.input_size = input_size
        self.code_size = code_size
        self.k = k
        self.lr = lr
        self.rng = random.Random(seed)
        self.encoder = [
            [(self.rng.random() * 2.0 - 1.0) / math.sqrt(input_size) for _ in range(input_size)]
            for _ in range(code_size)
        ]
        self.decoder = [
            [(self.rng.random() * 2.0 - 1.0) / math.sqrt(code_size) for _ in range(input_size)]
            for _ in range(code_size)
        ]

    def encode(self, x: List[float]) -> List[float]:
        raw = [math.tanh(dot(row, x)) for row in self.encoder]
        active = set(top_k_abs(raw, self.k))
        return [raw[i] if i in active else 0.0 for i in range(self.code_size)]

    def decode(self, code: List[float]) -> List[float]:
        rec = [0.0 for _ in range(self.input_size)]
        for i, c in enumerate(code):
            if abs(c) <= 1e-12:
                continue
            row = self.decoder[i]
            for j in range(self.input_size):
                rec[j] += c * row[j]
        return rec

    def train_step(self, x: List[float]) -> SparseCode:
        code = self.encode(x)
        rec = self.decode(code)
        err = [target - got for target, got in zip(x, rec)]
        loss = mse(x, rec)
        active = [i for i, c in enumerate(code) if abs(c) > 1e-12]

        for i in active:
            c = code[i]
            row = self.decoder[i]
            for j in range(self.input_size):
                row[j] += self.lr * err[j] * c
            # A small Hebbian nudge lets the random encoder adapt without
            # turning this into full backprop.
            enc = self.encoder[i]
            for j in range(self.input_size):
                enc[j] += self.lr * 0.04 * err[j] * x[j] * (1.0 if c >= 0 else -1.0)
            normalize_in_place(enc, target=1.0)

        return SparseCode(code=code, reconstruction=rec, loss=loss, active=active)

    def to_dict(self) -> dict:
        return {
            "input_size": self.input_size,
            "code_size": self.code_size,
            "k": self.k,
            "lr": self.lr,
            "encoder": self.encoder,
            "decoder": self.decoder,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "OnlineSparseAutoencoder":
        obj = cls(
            input_size=int(data.get("input_size", 96)),
            code_size=int(data.get("code_size", 32)),
            k=int(data.get("k", 6)),
            lr=float(data.get("lr", 0.035)),
        )
        obj.encoder = [[float(x) for x in row] for row in data.get("encoder", obj.encoder)]
        obj.decoder = [[float(x) for x in row] for row in data.get("decoder", obj.decoder)]
        return obj
