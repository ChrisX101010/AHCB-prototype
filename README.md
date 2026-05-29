# AHCB-0 Prototype

AHCB-0 is a tiny local prototype for the "Artificial Human Cognitive Brain"
idea: an autonomous, self-pruning cognitive cube that learns inside a dynamic
toy world.

It is deliberately **not** an LLM wrapper. It uses only Python's standard
library:

- omnidirectional sandbox sensors inspired by Argus
- echo-state reservoir dynamics
- sparse autoencoder-style compression
- 3D cognitive cube memory
- Rubik/Stockfish-style solver moves over internal state
- consolidation and pruning
- objective tests instead of LLM-as-judge

Run:

```bash
python run_demo.py
python tests.py
python simulate.py --steps 80
python benchmark.py
python train.py --steps 5000
```

The first useful question is not "is this intelligent yet?" It is:

> Can a small autonomous system learn a non-static world, compress experience,
> prune memory, and improve simple prediction/retrieval without cloud compute?

## Development Lab

Run benchmarks against random and reflex baselines:

```bash
python benchmark.py --levels 4 --seeds 5 --steps 220 --train-steps 700
```

Run autonomous local training:

```bash
python train.py --steps 5000 --chunk 500 --save runs/ahcb_brain.json
```

For an overnight run, increase the step count:

```bash
python train.py --steps 100000 --chunk 1000 --save runs/ahcb_brain.json
```

Interact with the sandbox:

```bash
python interact.py
```

## Visual Simulation

```bash
python simulate.py --steps 80
python simulate.py --steps 120 --live --delay 0.05
```

Legend:

- `A` = agent
- `G` = goal
- `#` = obstacle
- `!` = hazard
- `+` = boundary
