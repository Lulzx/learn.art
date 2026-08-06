# Roadmap

- v0.1: dense tensors, tensor autograd, named graphs, SGD.
- v0.2: runtime inputs and ordinary graph data.
- v0.3: linear and logistic regression.
- v0.4: neural activations, stateful optimizers, dense MLPs.
- v0.5: graph validation, pruning, duplicate discovery, DOT export.
- v0.6: constant folding, CSE aliases, fusion schedules, CPU execution.
- v0.7: optional compiled CPU fusion kernels with reference fallback.

## Next

The next coherent milestone is tensor generalization: ranks above two, explicit axis reductions, shape inference, and view-safe indexing. GPU and device work should follow only after those semantics are shared by eager, reference, and native CPU execution.
