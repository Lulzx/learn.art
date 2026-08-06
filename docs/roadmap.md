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
- v0.14: resumable high-level training with named feeds, validation, history, and callback stopping.
- v0.15: global gradient clipping and checkpointable epoch learning-rate schedules.
- v0.16: fitted standardization, reversible transform pipelines, and portable preprocessing state.
- v0.17: stable softmax and cross entropy, shaped inputs, class prediction, and multiclass accuracy.

## Next

The next coherent milestone is regularization: dropout with explicit train/eval mode and parameter penalties that compose with graph training.
