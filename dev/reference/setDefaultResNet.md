# Create Default ResNet Settings

Deprecated:
[`setResNet()`](https://ohdsi.github.com/DeepPatientLevelPrediction/dev/reference/setResNet.md)
now defaults to the same model.

## Usage

``` r
setDefaultResNet(
  estimatorSettings = setEstimator(learningRate = "auto", weightDecay = 1e-06, device =
    "cpu", batchSize = 1024, epochs = 50, seed = NULL)
)
```

## Arguments

- estimatorSettings:

  Estimator settings created by
  [`setEstimator()`](https://ohdsi.github.com/DeepPatientLevelPrediction/dev/reference/setEstimator.md).

## Value

A `modelSettings` object for use with `PatientLevelPrediction`.
