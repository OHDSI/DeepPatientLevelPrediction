# Create default settings for a non-temporal transformer

Creates settings for the package's default non-temporal transformer
model.

## Usage

``` r
setDefaultTransformer(
  estimatorSettings = setEstimator(learningRate = "auto", weightDecay = 1e-04, batchSize
    = 512, epochs = 10, seed = NULL, device = "cpu")
)
```

## Arguments

- estimatorSettings:

  Estimator settings created by
  [`setEstimator()`](https://ohdsi.github.com/DeepPatientLevelPrediction/reference/setEstimator.md).

## Value

A `modelSettings` object for use with `PatientLevelPrediction`.

## Details

The architecture and default hyperparameters are based on [Gorishniy et
al. (2021)](https://arxiv.org/abs/2106.11959).

## Examples

``` r
transformerSettings <- setDefaultTransformer()
transformerSettings$param[[1]]$numBlocks
#> [1] 3
```
