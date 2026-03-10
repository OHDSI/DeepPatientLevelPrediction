# Create default settings for a non-temporal transformer

A transformer model with default hyperparameters

## Usage

``` r
setDefaultTransformer(
  estimatorSettings = setEstimator(learningRate = "auto", weightDecay = 1e-04, batchSize
    = 512, epochs = 10, seed = NULL, device = "cpu")
)
```

## Arguments

- estimatorSettings:

  created with \`setEstimator\`

## Details

from https://arxiv.org/abs/2106.11959 Default hyperparameters from paper
