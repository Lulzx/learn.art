# Roadmap

- v0.1: dense tensors, tensor autograd, named graphs, SGD.
- v0.2: runtime inputs and ordinary graph data.
- v0.3: linear and logistic regression.
- v0.4: neural activations, stateful optimizers, dense MLPs.
- v0.5: graph validation, pruning, duplicate discovery, DOT export.
- v0.6: constant folding, CSE aliases, fusion schedules, CPU execution.
- v0.7: optional compiled CPU fusion kernels with reference fallback.
- v0.8: arbitrary-rank tensors, axis reductions, axis permutations, and safe scalar indexing.
- v0.9: explicit CPU placement, differentiable transfers, and a versioned tensor-buffer contract.

## Next

The next coherent milestone is model state: named state dictionaries, strict parameter restoration, and portable checkpoints built on the tensor-buffer contract.
