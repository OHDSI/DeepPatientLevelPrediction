# Create ResNet Settings

Creates model and hyperparameter-search settings for a residual network.

## Usage

``` r
setResNet(
  numLayers = c(1:8),
  sizeHidden = c(2^(6:10)),
  hiddenFactor = c(1:4),
  residualDropout = c(seq(0, 0.5, 0.05)),
  hiddenDropout = c(seq(0, 0.5, 0.05)),
  sizeEmbedding = c(2^(6:9)),
  estimatorSettings = setEstimator(learningRate = "auto", weightDecay = c(1e-06, 0.001),
    device = "cpu", batchSize = 1024, epochs = 30, seed = NULL),
  hyperParamSearch = "random",
  randomSample = 100,
  randomSampleSeed = NULL
)
```

## Arguments

- numLayers:

  Number of residual layers.

- sizeHidden:

  Width of the hidden representation.

- hiddenFactor:

  Multiplier controlling the inner width of each residual layer.

- residualDropout:

  Dropout probability after the final linear operation in each residual
  layer.

- hiddenDropout:

  Dropout probability after the first linear operation in each residual
  layer.

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

## Details

The architecture is based on [Gorishniy et al.
(2021)](https://arxiv.org/abs/2106.11959).

## Examples

``` r
resnetSettings <- setResNet(
  numLayers = c(2, 4),
  sizeHidden = 128,
  hiddenFactor = 2,
  residualDropout = 0.1,
  hiddenDropout = 0.1,
  sizeEmbedding = 64,
  randomSample = 2,
  randomSampleSeed = 42
)
```
