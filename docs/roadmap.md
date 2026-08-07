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
- v0.18: seeded inverted dropout, explicit train/eval mode, resumable regularization state, and L1/L2 penalties.
- v0.19: reusable dense layers, nested parameter discovery, named groups, and resumable selective freezing.
- v0.20: local seeded Xavier/He/uniform initialization, reset APIs, and repeatable initializer metadata.
- v0.21: confusion matrices, per-class and macro reports, and mergeable streaming classification meters.
- v0.22: immutable executable inference snapshots with optional preprocessing and a safe portable artifact codec.
- v0.23: named input gradients, indexed saliency, and baseline-integrated gradients with state restoration.
- v0.24: pluggable device backends and reusable compiled MPSGraph inference through `arturo-metal`.
- v0.25: GPU-resident MLP training, direct IDX-backed MNIST execution, and NCHW convolution/pooling schedules.
- v0.26: ordinary-data differentiation and optimizer graph transforms compiled as generic MPSGraph training programs.
- v0.27: eager convolution/pooling reverse kernels and generic compiled convolution training on MPSGraph.

## Next

The next coherent milestone is lazy host materialization for GPU tensors and training results.
