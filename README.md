# learn.art

`learn` is differentiable programming as ordinary Arturo data. Version 0.5 combines a small tensor/autograd runtime with inspectable models, neural composition, and graph-data transformations.

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
- `shape`, `reshape`, `transpose`, `square`, `tensorSum`, `mean`, and `matmul`

Tensors contain floating-point `data`, `shape`, and row-major `strides`. Only ranks 0–2 are accepted. Broadcasting aligns trailing dimensions and supports scalar, vector, and matrix operands. `tensorSum` is named explicitly because Arturo already owns the global `sum` word for blocks and ranges.

## Autograd and models

`variable`, `backward`, `gradient`, `detach`, and `zeroGrad` form the autograd vocabulary. Scalar values are promoted automatically, so the smallest proof starts with `x: variable 2.0`.

A graph evaluates its labeled block once to establish the operation DAG. `input` declares a placeholder and `forward.with:` supplies its runtime value. Graph fields are available dynamically (`model\loss`, `model\w`). `inputs`, `parameters`, and `operations` inspect its roles, while `graphData` returns a plain dictionary containing the graph order, nodes, shapes, operations, and parent relationships. `explain` renders that data for people.

`sgd.rate: 0.05 (parameters model)` creates a stateful optimizer. `step` updates parameter tensor data while preserving graph identity.

## Regression API

`fit.linear features targets` and `fit.logistic features targets` build and train ordinary learn graphs. Use `predict`, `predict.probability`, `score`, `mse`, and `accuracy` to evaluate them. `split.ratio:` performs a deterministic paired row split. The first estimator layer accepts vectors or one-column matrices.

`fit.mlp.hidden: 4 features targets` builds a dense one-hidden-layer regression graph. `relu`, `tensorTanh`, `dense`, `momentum`, and `adam` are also public building blocks.

## Graph transformations

`validateGraphData`, `deadCodeEliminate`, `commonSubexpressions`, and `optimizeGraph` analyze or rewrite the plain dictionary returned by `graphData`. `graphDot` exports the same representation for Graphviz visualization. These passes preserve the live eager graph and make lazy execution work explicit at the data boundary.

## Development

The package requires Arturo 0.10.0 and uses [unitt](https://github.com/RickBarretto/unitt):

```sh
arturo -p install unitt
~/.arturo/packages/bin/unitt --no-color
arturo examples/linear.art
arturo examples/regression.art
```

GPU execution, ranks above two, mixed dtypes, views, mutation APIs, and native backends remain outside the current milestone.
