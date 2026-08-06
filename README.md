# learn.art

`learn` is differentiable programming as ordinary Arturo data. Version 0.13 combines arbitrary-rank, device-aware tensors and autograd with generalized linear algebra, deterministic mini-batching, resumable training sessions, executable schedules, and optional native CPU fusion kernels.

## Example

```arturo
import "learn"!

do [
    xTrain: tensor [0.0 1.0 2.0 3.0]
    yTrain: tensor [1.0 3.0 5.0 7.0]

    model: graph [
        x: input
        y: input
        w: parameter 0.0
        b: parameter 0.0
        linear: x * w
        prediction: linear + b
        residual: prediction - y
        squared: square residual
        loss: mean squared
    ]

    optimizer: sgd.rate: 0.05 (parameters model)

    do.times: 500 [
        zeroGrad model
        forward.with: #[x: xTrain y: yTrain] model
        backward model\loss
        step optimizer
    ]

    print model\w
    print model\b
]
```

The result converges to `w ≈ 2` and `b ≈ 1`. Named intermediate expressions are deliberate: Arturo has no operator precedence, and explicit graph steps keep both evaluation and graph inspection unambiguous.

## Tensor API

- `tensor value`, `tensor.zeros shape`, `tensor.ones shape`, and `tensor.random shape`
- overloaded `+`, `-`, `*`, `/`, `^`, and unary `neg`
- `shape`, `reshape`, `transpose`, `tensorAt`, `square`, `tensorSum`, `mean`, and `matmul`
- `availableDevices`, `deviceOf`, `toDevice`, `tensorBuffer`, and `tensorFromBuffer`

Tensors contain owned floating-point `data`, inferred `shape`, and row-major `strides`. Rectangular nested blocks may have any rank, and broadcasting aligns trailing dimensions. `tensorSum.axis:` and `mean.axis:` reduce one explicit axis (negative axes count from the end); without `axis:` they reduce the whole tensor. `transpose.axes:` accepts a full axis permutation, while plain `transpose` swaps the last two axes. `tensorAt value [i j ...]` returns an owned scalar copy and supports negative indices.

`matmul` follows the usual generalized rules: two vectors produce a scalar dot product, matrix–vector and vector–matrix products remove the promoted unit dimension, and rank-2-or-higher operands broadcast their leading batch dimensions. The same shapes and batch accumulation rules apply to reverse-mode gradients and scheduled CPU execution.

Every tensor carries an explicit device. v0.9 provides the CPU device, `tensor.device: 'cpu`, differentiable `toDevice`, and an owned `learn.buffer`/`float64` interchange contract. Device metadata survives graph export, optimization, and scheduled execution; unsupported devices fail at the placement boundary.

## Autograd and models

`variable`, `backward`, `gradient`, `detach`, and `zeroGrad` form the autograd vocabulary. Scalar values are promoted automatically, so the smallest proof starts with `x: variable 2.0`.

A graph evaluates its labeled block once to establish the operation DAG. `input` declares a placeholder and `forward.with:` supplies its runtime value. Graph fields are available dynamically (`model\loss`, `model\w`). `inputs`, `parameters`, and `operations` inspect its roles, while `graphData` returns a plain dictionary containing the graph order, nodes, shapes, operations, and parent relationships. `explain` renders that data for people.

`sgd.rate: 0.05 (parameters model)` creates a stateful optimizer. `step` updates parameter tensor data while preserving graph identity.

`stateDict model` returns an owned, named parameter snapshot built from tensor buffers. `loadStateDict model state` validates the complete snapshot before mutating anything and is strict by default; use `loadStateDict.strict: false` for intentional partial restoration. `saveCheckpoint model path` and `loadCheckpoint model path` round-trip the same state through a portable Arturo-text format that is parsed as data rather than executed.

`optimizerState` and `loadOptimizerState` preserve SGD rates, Momentum velocities, and Adam hyperparameters, moments, and timestep. `saveTrainingCheckpoint model optimizer path` and `loadTrainingCheckpoint model optimizer path` combine model and optimizer state; both halves are fully validated before either live object is changed.

`batcher.size:.seed:` creates a deterministic paired feature/target iterator. `nextBatch` returns owned batch tensors, original row indices, and the current epoch. `batcherState` restores the exact permutation and cursor, while `saveSessionCheckpoint` and `loadSessionCheckpoint` atomically combine model, optimizer, and data-order state.

## Regression API

`fit.linear features targets` and `fit.logistic features targets` build and train ordinary learn graphs. Use `predict`, `predict.probability`, `score`, `mse`, and `accuracy` to evaluate them. `split.ratio:` performs a deterministic paired row split. The first estimator layer accepts vectors or one-column matrices.

`fit.mlp.hidden: 4 features targets` builds a dense one-hidden-layer regression graph. `relu`, `tensorTanh`, `dense`, `momentum`, and `adam` are also public building blocks.

## Graph transformations

`validateGraphData`, `deadCodeEliminate`, `commonSubexpressions`, and `optimizeGraph` analyze or rewrite the plain dictionary returned by `graphData`. `graphDot` exports the same representation for Graphviz visualization. These passes preserve the live eager graph and make lazy execution work explicit at the data boundary.

`scheduleGraph data 'loss` applies constant folding, alias-preserving CSE, output pruning, and elementwise fusion planning. `executeCpu.with: feeds schedule` evaluates that immutable parameter snapshot using the pure-Arturo CPU reference backend.

`compileNativeCpu schedule` lowers eligible fusion groups to generated C when `clang` is available. `executeNativeCpu.with: feeds artifact` uses those kernels and automatically falls back to the reference backend for unsupported groups or dynamic shapes.

## Development

The package requires Arturo 0.10.0 and uses [unitt](https://github.com/RickBarretto/unitt):

```sh
arturo -p install unitt
~/.arturo/packages/bin/unitt --no-color
arturo examples/linear.art
arturo examples/regression.art
arturo examples/native-cpu.art
```

GPU execution, mixed dtypes, slice views, mutation APIs, and additional native backends remain outside the current milestone.
