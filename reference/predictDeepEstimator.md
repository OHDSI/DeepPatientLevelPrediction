# Predict with a Deep Learning Estimator

Applies a fitted deep learning model to prepared patient-level
prediction data.

## Usage

``` r
predictDeepEstimator(plpModel, data, cohort)
```

## Arguments

- plpModel:

  A fitted `plpModel` or an in-memory Python estimator.

- data:

  A `plpData` object or a compatible PyTorch dataset.

- cohort:

  A data frame containing the row identifiers to score.

## Value

A data frame containing the input cohort columns and predicted values in
the `value` column.

## Examples

``` r
if (FALSE) { # \dontrun{
# Requires FeatureExtraction and the package's Python dependencies,
# including PyTorch.
# See vignette("Installing") for setup instructions.
data("simulationProfile", package = "PatientLevelPrediction")
plpData <- PatientLevelPrediction::simulatePlpData(
  simulationProfile, n = 200, seed = 42
)
# Supply metadata omitted by simulatePlpData() for this bundled profile:
# gender, age, conditions, and drugs. Only age is continuous.
plpData$covariateData$analysisRef <- data.frame(
  analysisId = c(1, 2, 102, 402),
  isBinary = c("Y", "N", "Y", "Y"),
  missingMeansZero = c(NA, "Y", NA, NA)
)
population <- PatientLevelPrediction::createStudyPopulation(
  plpData,
  populationSettings = PatientLevelPrediction::createStudyPopulationSettings(
    riskWindowEnd = 90, minTimeAtRisk = 89
  )
)
splitData <- PatientLevelPrediction::splitData(
  plpData,
  population,
  splitSettings = PatientLevelPrediction::createDefaultSplitSetting(
    testFraction = 0.25, trainFraction = 0.75, nfold = 2, splitSeed = 42
  )
)

# Keep this toy example small: one configuration, two folds, and one epoch.
# These settings demonstrate fitting, not meaningful predictive performance.
modelSettings <- setResNet(
  numLayers = 1,
  sizeHidden = 8,
  hiddenFactor = 1,
  residualDropout = 0,
  hiddenDropout = 0,
  sizeEmbedding = 8,
  hyperParamSearch = "grid",
  estimatorSettings = setEstimator(
    learningRate = 0.001, batchSize = 64, epochs = 1, seed = 42
  )
)
analysisPath <- tempfile("deep-plp-example-")
dir.create(analysisPath)
model <- fitEstimator(
  trainData = splitData$Train,
  modelSettings = modelSettings,
  analysisId = 1,
  analysisPath = analysisPath
)

# Score held-out test patients using only the fitted training model.
prediction <- predictDeepEstimator(
  plpModel = model,
  data = splitData$Test,
  cohort = splitData$Test$labels
)
head(prediction)

# Clean up after inspecting predictions.
Andromeda::close(splitData$Test$covariateData)
Andromeda::close(splitData$Train$covariateData)
Andromeda::close(plpData$covariateData)
unlink(analysisPath, recursive = TRUE)
unlink(model$model, recursive = TRUE)
} # }
```
