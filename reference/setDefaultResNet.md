# Create Default ResNet Settings

Creates settings for the package's default residual-network model.

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
  [`setEstimator()`](https://ohdsi.github.com/DeepPatientLevelPrediction/reference/setEstimator.md).

## Value

A `modelSettings` object for use with `PatientLevelPrediction`.

## Details

The architecture is based on [Gorishniy et al.
(2021)](https://arxiv.org/abs/2106.11959). The hyperparameters are
defaults selected for patient-level prediction tasks.

## Examples

``` r
resnetSettings <- setDefaultResNet()
resnetSettings$param[[1]]$numLayers
#> [1] 6
```
