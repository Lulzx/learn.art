# Roadmap

- v0.1: dense tensors, tensor autograd, named graphs, SGD.
- v0.2: runtime inputs and ordinary graph data.
- v0.3: linear and logistic regression.
- v0.4: neural activations, stateful optimizers, dense MLPs.
- v0.5: graph validation, pruning, duplicate discovery, DOT export.
- v0.6: constant folding, CSE aliases, fusion schedules, CPU execution.
- v0.7: optional compiled CPU fusion kernels with reference fallback.
- v0.8: arbitrary-rank tensors, axis reductions, axis permutations, and safe scalar indexing.

## Next

The next coherent milestone is device-aware storage: a backend-neutral tensor buffer contract, explicit device placement, transfer semantics, and CPU parity fixtures that can later admit GPU execution without changing tensor meaning.
