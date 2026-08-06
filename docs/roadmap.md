# Roadmap

- v0.1: dense tensors, tensor autograd, named graphs, SGD.
- v0.2: runtime inputs and ordinary graph data.
- v0.3: linear and logistic regression.
- v0.4: neural activations, stateful optimizers, dense MLPs.
- v0.5: graph validation, pruning, duplicate discovery, DOT export.
- v0.6: constant folding, CSE aliases, fusion schedules, CPU execution.

## Next

The next coherent milestone is native kernel lowering: benchmark the v0.6 reference schedule, define a stable backend ABI, lower fused groups to compiled CPU kernels, and compare every result against `executeCpu`. Higher-rank tensors, views, devices, and GPU kernels should follow only after that path is measurable and correct.
