# Create default settings for a non-temporal transformer

Deprecated:
[`setTransformer()`](https://ohdsi.github.com/DeepPatientLevelPrediction/dev/reference/setTransformer.md)
now defaults to the same model.

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
  [`setEstimator()`](https://ohdsi.github.com/DeepPatientLevelPrediction/dev/reference/setEstimator.md).

## Value

A `modelSettings` object for use with `PatientLevelPrediction`.
