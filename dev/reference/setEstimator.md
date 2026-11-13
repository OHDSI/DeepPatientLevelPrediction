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
  trainValidationSplit = FALSE,
  initOutputBias = TRUE,
  gradClipNorm = NULL,
  deterministic = FALSE
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

  Early-stopping settings: a list with `useEarlyStopping` and the
  `params` `patience`, `delta` and `min_epochs`. `NULL` or
  `useEarlyStopping = FALSE` disable early stopping.

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

- initOutputBias:

  Whether to start the bias of the output layer at the log-odds of the
  outcome rate in the training data, so that the first predictions match
  the outcome rate instead of 50%. Without it, models for rare outcomes
  spend their first epochs lowering all predictions, and early stopping
  can end training before they learn to rank patients. It applies to
  models whose output is a single logit from a linear `head` layer and
  is not used during the learning-rate search.

- gradClipNorm:

  The largest norm the gradients may have at each optimizer step, or
  `NULL` to leave them as they are. Larger gradients are scaled down to
  this norm, which keeps a few extreme batches from destabilizing
  training. The loss is summed over the patients of a batch, so the
  gradient norm grows with the batch size. Several values are tuned as
  hyperparameters.

- deterministic:

  Whether PyTorch may only use deterministic algorithms, so that
  training with the same seed and data gives the same model each time.
  This matters mostly on a GPU, and makes training slower. Operations
  that have no deterministic version give a warning.

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
