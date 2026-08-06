# Lazy IR

The lazy pipeline is deliberately separate from live autograd objects:

```text
graphData → constantFold → CSE aliases → DCE → fusionGroups → scheduleGraph → executeCpu
```

Alias nodes preserve named CSE results. Fusion groups are scheduling hints; they do not change numerical semantics. The v0.6 CPU backend interprets scheduled tensor operations and establishes the contract future native kernels must match.
