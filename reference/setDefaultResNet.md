# setDefaultResNet

Creates settings for a default ResNet model

## Usage

``` r
setDefaultResNet(
  estimatorSettings = setEstimator(learningRate = "auto", weightDecay = 1e-06, device =
    "cpu", batchSize = 1024, epochs = 50, seed = NULL)
)
```

## Arguments

- estimatorSettings:

  created with “\`setEstimator“\`

## Details

Model architecture from by https://arxiv.org/abs/2106.11959 .
Hyperparameters chosen by a experience on a few prediction problems.
