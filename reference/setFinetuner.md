# Create Fine-Tuning Settings

Creates settings for fine-tuning a previously fitted deep learning
model.

## Usage

``` r
setFinetuner(modelPath, estimatorSettings = setEstimator())
```

## Arguments

- modelPath:

  Path to an existing saved `plpModel` directory.

- estimatorSettings:

  Estimator settings created by
  [`setEstimator()`](https://ohdsi.github.com/DeepPatientLevelPrediction/reference/setEstimator.md).

## Value

A `modelSettings` object that initializes from the saved model.

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
    testFraction = 0, trainFraction = 1, nfold = 2, splitSeed = 42
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

# Save a real fitted model before creating settings for a new training task.
modelPath <- file.path(analysisPath, "savedModel")
PatientLevelPrediction::savePlpModel(model, modelPath)
finetuneSettings <- setFinetuner(
  modelPath = modelPath,
  estimatorSettings = setEstimator(
    learningRate = 0.001, batchSize = 64, epochs = 1, seed = 42
  )
)
finetuneSettings$modelType

# Clean up this example's files. In real use, keep modelPath until
# fine-tuning finishes: finetuneSettings refers to the saved model files.
Andromeda::close(splitData$Train$covariateData)
Andromeda::close(plpData$covariateData)
unlink(analysisPath, recursive = TRUE)
unlink(model$model, recursive = TRUE)
} # }
```
