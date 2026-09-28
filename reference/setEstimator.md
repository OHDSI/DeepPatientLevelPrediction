# Create Estimator Settings

Creates settings controlling optimization, model fitting, and model
selection for a deep learning estimator.

## Usage

``` r
setEstimator(
  learningRate = "auto",
  weightDecay = 0,
  batchSize = 512,
  epochs = 30,
  device = "cpu",
  optimizer = torch$optim$AdamW,
  scheduler = list(fun = torch$optim$lr_scheduler$ReduceLROnPlateau, params =
    list(patience = 1)),
  criterion = torch$nn$BCEWithLogitsLoss,
  earlyStopping = list(useEarlyStopping = TRUE, params = list(patience = 4)),
  compile = FALSE,
  metric = "auc",
  accumulationSteps = NULL,
  seed = NULL,
  trainValidationSplit = FALSE
)
```

## Arguments

- learningRate:

  Learning rate, or `"auto"` to use the learning-rate finder.

- weightDecay:

  Weight-decay value.

- batchSize:

  Number of observations per batch.

- epochs:

  Maximum number of training epochs.

- device:

  Device on which to train. This can be a string or a function evaluated
  when training begins.

- optimizer:

  PyTorch optimizer constructor. Evaluation is delayed until training
  begins.

- scheduler:

  A list containing the learning-rate scheduler constructor in `fun` and
  its arguments in `params`. Evaluation is delayed until training
  begins.

- criterion:

  PyTorch loss constructor. Evaluation is delayed until training begins.

- earlyStopping:

  Early-stopping settings, or `NULL` to disable early stopping.

- compile:

  Whether to compile the PyTorch model before training.

- metric:

  Either `"auc"`, `"loss"`, or a list defining a custom metric. A custom
  metric list must contain `fun`, `mode`, and `name`.

- accumulationSteps:

  Number of batches over which to accumulate gradients, or a function
  evaluated when training begins.

- seed:

  Random seed used to initialize the model. A seed is generated when
  this is `NULL`.

- trainValidationSplit:

  Whether to use a train-validation split for model selection instead of
  cross-validation.

## Value

A list of estimator settings used by the model-setting functions.

## Examples

``` r
estimatorSettings <- setEstimator(
  learningRate = 0.001,
  batchSize = 128,
  epochs = 10,
  seed = 42
)
estimatorSettings$batchSize
#> [1] 128
```
