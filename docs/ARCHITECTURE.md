# AHCB-0 Architecture

AHCB-0 is a small experimental cognitive substrate. It is not an LLM, not a
chatbot, and not a cloud service. The point is to test whether a local system
can autonomously sense, move, compress, consolidate, prune, and improve simple
behavior in a changing world.

## Loop

```text
observe omnidirectionally
  -> reservoir dynamics
  -> sparse feature compression
  -> route into cognitive cube
  -> choose action
  -> observe reward/surprise
  -> solver performs internal cube moves
  -> consolidate center
  -> prune weak memories
```

## Modules

- `DynamicSandboxWorld`: a moving toy environment with obstacles, hazards, and goals.
- `EchoStateReservoir`: fixed recurrent dynamics that create temporal traces.
- `OnlineSparseAutoencoder`: k-sparse feature compressor with local online updates.
- `CognitiveCube`: 3D memory organized by region, modality, and abstraction level.
- `CubeSolver`: Rubik/Stockfish-inspired internal move selector.
- `AHCB0Agent`: autonomous scheduler and policy.

## What The Rubik Cube Means Here

The cube is not a literal puzzle. It is a state-space metaphor implemented as
addressable memory. Rubik-like moves are transformations of internal state:

- lift raw surprise into pattern memory
- rotate a region's modality routing
- consolidate local cells into a center schema
- prune old low-utility cells and links

## What Argus Contributes

Argus suggests that useful intelligence should not have one privileged front.
AHCB-0 copies that design pressure with eight-direction sensing. Local hazard,
wall, obstacle, and goal signals can influence action before slower learned
values do.

## What This Prototype Can Prove

It can test:

- memory formation
- reconstruction loss from sparse compression
- local action learning
- bounded memory through pruning
- internal solver activity
- adaptation to a dynamic sandbox

It cannot yet prove:

- human-level cognition
- language fluency
- real multimodal understanding
- replacement of backprop on frontier-scale models

