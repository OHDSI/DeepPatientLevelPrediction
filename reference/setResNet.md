# setResNet

Creates settings for a ResNet model

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

  Number of layers in network, default: 1:16

- sizeHidden:

  Amount of neurons in each default layer, default: 2^(6:10) (64 to
  1024)

- hiddenFactor:

  How much to grow the amount of neurons in each ResLayer, default: 1:4

- residualDropout:

  How much dropout to apply after last linear layer in ResLayer,
  default: seq(0, 0.3, 0.05)

- hiddenDropout:

  How much dropout to apply after first linear layer in ResLayer,
  default: seq(0, 0.3, 0.05)

- sizeEmbedding:

  Size of embedding layer, default: 2^(6:9) '(64 to 512)

- estimatorSettings:

  created with “\`setEstimator“\`

- hyperParamSearch:

  Which kind of hyperparameter search to use random sampling or
  exhaustive grid search. default: 'random'

- randomSample:

  How many random samples from hyperparameter space to use

- randomSampleSeed:

  Random seed to sample hyperparameter combinations

## Details

Model architecture from by https://arxiv.org/abs/2106.11959
