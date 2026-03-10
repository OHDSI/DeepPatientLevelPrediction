# setEstimator

creates settings for the Estimator, which takes a model and trains it

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

  what learning rate to use

- weightDecay:

  what weight_decay to use

- batchSize:

  batchSize to use

- epochs:

  how many epochs to train for

- device:

  what device to train on, can be a string or a function to that
  evaluates to the device during runtime

- optimizer:

  which optimizer to use

- scheduler:

  which learning rate scheduler to use

- criterion:

  loss function to use

- earlyStopping:

  If earlyStopping should be used which stops the training of your
  metric is not improving

- compile:

  if the model should be compiled before training, default FALSE

- metric:

  either \`auc\` or \`loss\` or a custom metric to use. This is the
  metric used for scheduler and earlyStopping. Needs to be a list with
  function \`fun\`, mode either \`min\` or \`max\` and a \`name\`,
  \`fun\` needs to be a function that takes in prediction and labels and
  outputs a score.

- accumulationSteps:

  how many steps to accumulate gradients before updating weights, can
  also be a function that is evaluated during runtime

- seed:

  seed to initialize weights of model with

- trainValidationSplit:

  if TRUE, perform a train-validation split for model selection instead
  of cross validation
