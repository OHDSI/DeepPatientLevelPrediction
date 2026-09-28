# Create Multilayer Perceptron Settings

Creates model and hyperparameter-search settings for a multilayer
perceptron.

## Usage

``` r
setMultiLayerPerceptron(
  numLayers = c(1:8),
  sizeHidden = c(2^(6:9)),
  dropout = c(seq(0, 0.3, 0.05)),
  sizeEmbedding = c(2^(6:9)),
  estimatorSettings = setEstimator(learningRate = "auto", weightDecay = c(1e-06, 0.001),
    batchSize = 1024, epochs = 30, device = "cpu"),
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
  [`setEstimator()`](https://ohdsi.github.com/DeepPatientLevelPrediction/reference/setEstimator.md).

- hyperParamSearch:

  Hyperparameter-search strategy, either `"random"` or `"grid"`.

- randomSample:

  Number of combinations sampled when `hyperParamSearch = "random"`.

- randomSampleSeed:

  Random seed used when sampling combinations.

## Value

A `modelSettings` object for use with `PatientLevelPrediction`.

## Examples

``` r
mlpSettings <- setMultiLayerPerceptron(
  numLayers = c(1, 2),
  sizeHidden = 64,
  dropout = 0.1,
  sizeEmbedding = 32,
  randomSample = 2,
  randomSampleSeed = 42
)
```
