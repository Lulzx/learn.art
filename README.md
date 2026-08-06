# learn.art

`learn` is a small, pure-Arturo foundation for differentiable programs. Version 0.1 provides owned dense tensors, reverse-mode automatic differentiation, named model graphs, and stateful SGD. It intentionally stops before high-level estimators: the first job is to prove that Arturo can express and train a model correctly.

## Example

```arturo
import "learn"!

do [
    x: tensor [[0.0] [1.0] [2.0] [3.0]]
    y: tensor [[1.0] [3.0] [5.0] [7.0]]

    model: graph [
        w: parameter.random [1 1]
        b: parameter.zeros [1]
        linear: matmul x w
        prediction: linear + b
        residual: prediction - y
        squared: square residual
        loss: mean squared
    ]

    optimizer: sgd.rate: 0.05 (parameters model)

    do.times: 500 [
        zeroGrad model
        forward model
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

`variable`, `backward`, `gradient`, `detach`, and `zeroGrad` form the autograd vocabulary. A graph evaluates its labeled block once to establish the operation DAG; `forward` then recomputes every recorded operation from the current parameter values. Graph fields are available dynamically (`model\loss`, `model\w`), and `parameters model` returns parameters in declaration order.

`sgd.rate: 0.05 (parameters model)` creates a stateful optimizer. `step` updates parameter tensor data while preserving graph identity.

## Development

The package requires Arturo 0.10.0 and uses [unitt](https://github.com/RickBarretto/unitt):

```sh
arturo -p install unitt
~/.arturo/packages/bin/unitt --no-color
arturo examples/linear.art
```

GPU execution, ranks above two, mixed dtypes, views, mutation APIs, neural-network layers, high-level estimators, and non-differentiable algorithms are outside the 0.1 milestone.
