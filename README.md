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
```

The first useful question is not "is this intelligent yet?" It is:

> Can a small autonomous system learn a non-static world, compress experience,
> prune memory, and improve simple prediction/retrieval without cloud compute?

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

