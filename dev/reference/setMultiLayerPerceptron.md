# Create Multilayer Perceptron Settings

Creates model and hyperparameter-search settings for a multilayer
perceptron.

## Usage

``` r
setMultiLayerPerceptron(
  numLayers = 3,
  sizeHidden = 256,
  dropout = 0.1,
  sizeEmbedding = 256,
  estimatorSettings = setEstimator(learningRate = "auto", weightDecay = 1e-06, batchSize
    = 1024, epochs = 50, device = "cpu"),
  hyperParamSearch = "random",
  randomSample = 100,
  randomSampleSeed = NULL
)
```

## Arguments

- numLayers:

  Number of hidden layers.

- sizeHidden:

  Number of units in each hidden layer.

- dropout:

  Dropout probability.

- sizeEmbedding:

  Embedding dimension.

- estimatorSettings:

  Estimator settings created by
  [`setEstimator()`](https://ohdsi.github.com/DeepPatientLevelPrediction/dev/reference/setEstimator.md).

- hyperParamSearch:

  Hyperparameter-search strategy, either `"random"` or `"grid"`.

- randomSample:

  Number of combinations sampled when `hyperParamSearch = "random"`. If
  there are fewer combinations, each is used once.

- randomSampleSeed:

  Random seed used when sampling combinations.

## Value

A `modelSettings` object for use with `PatientLevelPrediction`.

## Details

The defaults define one model. To search hyperparameters, give several
values; the example shows the search space that the package used to
search by default.

## Examples

``` r
# one model, with the default hyperparameters
mlpSettings <- setMultiLayerPerceptron()

# a random search over 100 of the 1,792 combinations of this space
mlpSearch <- setMultiLayerPerceptron(
  numLayers = 1:8,
  sizeHidden = 2^(6:9),
  dropout = seq(0, 0.3, 0.05),
  sizeEmbedding = 2^(6:9),
  estimatorSettings = setEstimator(
    learningRate = "auto",
    weightDecay = c(1e-6, 1e-3),
    batchSize = 1024,
    epochs = 30
  ),
  hyperParamSearch = "random",
  randomSample = 100,
  randomSampleSeed = 42
)
```
