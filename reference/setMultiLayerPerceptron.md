# setMultiLayerPerceptron

Creates settings for a Multilayer perceptron model

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

  Number of layers in network, default: 1:8

- sizeHidden:

  Amount of neurons in each default layer, default: 2^(6:9) (64 to 512)

- dropout:

  How much dropout to apply after first linear, default: seq(0, 0.3,
  0.05)

- sizeEmbedding:

  Size of embedding default: 2^(6:9) (64 to 512)

- estimatorSettings:

  settings of Estimator created with \`setEstimator\`

- hyperParamSearch:

  Which kind of hyperparameter search to use random sampling or
  exhaustive grid search. default: 'random'

- randomSample:

  How many random samples from hyperparameter space to use

- randomSampleSeed:

  Random seed to sample hyperparameter combinations

## Details

Model architecture
