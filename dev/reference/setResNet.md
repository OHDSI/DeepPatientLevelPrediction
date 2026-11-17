# Create ResNet Settings

Creates model and hyperparameter-search settings for a residual network.

## Usage

``` r
setResNet(
  numLayers = 6,
  sizeHidden = 512,
  hiddenFactor = 2,
  residualDropout = 0.1,
  hiddenDropout = 0.4,
  sizeEmbedding = 256,
  estimatorSettings = setEstimator(learningRate = "auto", weightDecay = 1e-06, device =
    "cpu", batchSize = 1024, epochs = 50, seed = NULL),
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

The architecture is based on [Gorishniy et al.
(2021)](https://arxiv.org/abs/2106.11959).

The defaults define one model, selected for patient-level prediction
tasks. To search hyperparameters, give several values; the example shows
the search space that the package used to search by default.

## Examples

``` r
# one model, with the default hyperparameters
resnetSettings <- setResNet()

# a random search over 100 of the 154,880 combinations of this space
resnetSearch <- setResNet(
  numLayers = 1:8,
  sizeHidden = 2^(6:10),
  hiddenFactor = 1:4,
  residualDropout = seq(0, 0.5, 0.05),
  hiddenDropout = seq(0, 0.5, 0.05),
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
