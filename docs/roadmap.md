# Roadmap

- v0.1: dense tensors, tensor autograd, named graphs, SGD.
- v0.2: runtime inputs and ordinary graph data.
- v0.3: linear and logistic regression.
- v0.4: neural activations, stateful optimizers, dense MLPs.
- v0.5: graph validation, pruning, duplicate discovery, DOT export.

## Next

The next coherent milestone is executable lazy graphs: constant folding, semantics-preserving CSE, fusion groups, scheduling, then a CPU-native backend. Higher-rank tensors, views, devices, and GPU kernels should follow only after that pipeline is measurable and correct.
