# Architecture

`learn` has one execution path:

1. Dense tensors store arbitrary-rank floating-point values with explicit device and dtype identity. Device allocations use mutable reference-counted ownership records. Slice views retain the root allocation, compose strided projections, and use mutation versions to invalidate cached projections after alias writes. Capable backends realize reads natively; writes use overlap-safe snapshots and atomic copy-and-swap storage replacement.
2. Autograd nodes record tensor operations and local derivative rules.
3. Graphs name nodes, parameters, and inputs.
4. Optimizers update parameter data while preserving node identity.
5. Estimators construct and train ordinary graphs.

The public identity is the graph dialect. Tensor and autograd code are its runtime machinery, not a separate product surface.
