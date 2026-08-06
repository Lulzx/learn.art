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
- v0.10: named model state, atomic strict restoration, and portable checkpoints.
- v0.11: vector products and broadcasted batched matrix multiplication with gradients.
- v0.12: optimizer state and atomic model-plus-optimizer training checkpoints.
- v0.13: deterministic paired mini-batches and fully resumable training-session checkpoints.

## Next

The next coherent milestone is a high-level training loop: named graph feeds, epoch/batch accounting, loss history, validation passes, and callback-based early stopping over resumable sessions.
