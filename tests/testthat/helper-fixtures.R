# Offline fixtures built from PatientLevelPrediction's simulated data. They let
# tests that need realistic data, but no database, run without the Eunomia
# integration setup. The `sim*` objects are promises: nothing is built, and
# Python is not started, until a test first uses one, and each is built once.

simulatedPlpData <- function(n = 1000, seed = 42) {
  profileEnv <- new.env()
  utils::data("simulationProfile", package = "PatientLevelPrediction", envir = profileEnv)
  plpData <- PatientLevelPrediction::simulatePlpData(
    profileEnv$simulationProfile, n = n, seed = seed
  )
  # simulatePlpData() leaves out analysisRef: gender, age, conditions and
  # drugs, of which only age is continuous
  plpData$covariateData$analysisRef <- data.frame(
    analysisId = c(1, 2, 102, 402),
    isBinary = c("Y", "N", "Y", "Y"),
    missingMeansZero = c(NA, "Y", NA, NA)
  )
  plpData
}

simulatedPopulationSettings <- function() {
  PatientLevelPrediction::createStudyPopulationSettings(
    riskWindowEnd = 90, minTimeAtRisk = 89
  )
}

delayedAssign("simPlpData", {
  plpData <- simulatedPlpData()
  # squared age as a second numeric feature, as the integration setup adds
  plpData$covariateData$covariateRef <- plpData$covariateData$covariateRef %>%
    dplyr::rows_append(data.frame(
      covariateId = 2002, covariateName = "Squared age", analysisId = 2, conceptId = 0
    ), copy = TRUE)
  squaredAges <- plpData$covariateData$covariates %>%
    dplyr::filter(.data$covariateId == 1002) %>%
    dplyr::mutate(covariateId = 2002, covariateValue = .data$covariateValue^2)
  plpData$covariateData$covariates <- plpData$covariateData$covariates %>%
    dplyr::rows_append(squaredAges)
  plpData
})

delayedAssign("simPopulation", PatientLevelPrediction::createStudyPopulation(
  simPlpData, outcomeId = 3, populationSettings = simulatedPopulationSettings()
))

delayedAssign("simTrainData", PatientLevelPrediction::splitData(
  simPlpData,
  population = simPopulation,
  splitSettings = PatientLevelPrediction::createDefaultSplitSetting(splitSeed = 42)
))

delayedAssign("simMappedData", PatientLevelPrediction::MapIds(
  covariateData = simTrainData$Train$covariateData,
  cohort = simTrainData$Train$labels
))

delayedAssign("simDataset", createDataset(
  data = simMappedData,
  labels = simTrainData$Train$labels,
  plpModel = NULL
))

delayedAssign("simSmallDataset", torch$utils$data$Subset(
  simDataset,
  (1:round(length(simDataset) / 3))
))

delayedAssign("simModelSettings", setResNet(
  numLayers = 1, sizeHidden = 16, hiddenFactor = 1,
  residualDropout = c(0, 0.2), hiddenDropout = 0,
  sizeEmbedding = 16, hyperParamSearch = "random",
  randomSample = 2,
  setEstimator(
    epochs = 1,
    learningRate = 3e-4
  )
))

# a fitted model, also saved to simFitEstimatorPath/plpModel
delayedAssign("simFitEstimatorPath", {
  path <- file.path(tempdir(), "simFitEstimator")
  dir.create(path, showWarnings = FALSE)
  path
})

delayedAssign("simFitEstimatorResults", {
  results <- fitEstimator(
    simTrainData$Train,
    modelSettings = simModelSettings,
    analysisId = 1,
    analysisPath = simFitEstimatorPath
  )
  PatientLevelPrediction::savePlpModel(results, file.path(simFitEstimatorPath, "plpModel"))
  results
})
