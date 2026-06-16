testLoc <- normalizePath(tempdir())
path <- system.file("python", package = "DeepPatientLevelPrediction")

dplpDebugStage <- function(stage) {
  if (identical(Sys.getenv("DPLP_SETUP_TRACE"), "true")) {
    cat("DPLP_SETUP_STAGE:", stage, "\n")
  }
  if (identical(Sys.getenv("DPLP_SETUP_STOP_AFTER"), stage)) {
    cat("DPLP_SETUP_STOP_AFTER:", stage, "\n")
    quit(save = "no", status = 0, runLast = TRUE)
  }
}

# get connection and data from Eunomia
connectionDetails <- Eunomia::getEunomiaConnectionDetails()
Eunomia::createCohorts(connectionDetails)
dplpDebugStage("cohorts")

covSet <- FeatureExtraction::createCovariateSettings(
  useDemographicsGender = TRUE,
  useDemographicsAge = TRUE,
  useDemographicsRace = TRUE,
  useDemographicsEthnicity = TRUE,
  useDemographicsAgeGroup = TRUE,
  useConditionGroupEraLongTerm = TRUE,
  useDrugEraStartLongTerm = TRUE,
  endDays = -1
)

tempCovSet <- FeatureExtraction::createTemporalSequenceCovariateSettings(
  useDemographicsGender = TRUE,
  useDemographicsAge = TRUE,
  useConditionOccurrence = TRUE,
  sequenceStartDay = -365,
  sequenceEndDay = -1
)

databaseDetails <- PatientLevelPrediction::createDatabaseDetails(
  connectionDetails = connectionDetails,
  cdmDatabaseSchema = "main",
  cohortDatabaseSchema = "main",
  cohortTable = "cohort",
  targetId = 4,
  outcomeIds = 3,
  outcomeDatabaseSchema = "main",
  outcomeTable = "cohort",
  cdmDatabaseName = "eunomia"
)

restrictPlpDataSettings <-
  PatientLevelPrediction::createRestrictPlpDataSettings(
    firstExposureOnly = TRUE,
    washoutPeriod = 365
  )

plpData <- PatientLevelPrediction::getPlpData(
  databaseDetails = databaseDetails,
  restrictPlpDataSettings = restrictPlpDataSettings,
  covariateSettings = covSet
)

plpDataTemporal <- PatientLevelPrediction::getPlpData(
  databaseDetails = databaseDetails,
  restrictPlpDataSettings = restrictPlpDataSettings,
  covariateSettings = tempCovSet
)
dplpDebugStage("plp-data")

# add age squared so I have more than one numerical feature
plpData$covariateData$covariateRef <- plpData$covariateData$covariateRef %>%
  dplyr::rows_append(data.frame(
    covariateId = 2002,
    covariateName = "Squared age",
    analysisId = 2,
    conceptId = 0
  ), copy = TRUE)

squaredAges <- plpData$covariateData$covariates %>%
  dplyr::filter(covariateId == 1002) %>%
  dplyr::mutate(
    covariateId = 2002,
    covariateValue = .data$covariateValue**2
  )

plpData$covariateData$covariates <- plpData$covariateData$covariates %>%
  dplyr::rows_append(squaredAges)

populationSet <- PatientLevelPrediction::createStudyPopulationSettings(
  requireTimeAtRisk = FALSE,
  riskWindowStart = 1,
  riskWindowEnd = 365
)

population <- PatientLevelPrediction::createStudyPopulation(
  plpData = plpData,
  outcomeId = 3,
  populationSettings = populationSet
)

trainData <- PatientLevelPrediction::splitData(
  plpData,
  population = population,
  splitSettings = PatientLevelPrediction::createDefaultSplitSetting(splitSeed = 42)
)

trainDataTemporal <- PatientLevelPrediction::splitData(
  plpDataTemporal,
  population = population,
  splitSettings = PatientLevelPrediction::createDefaultSplitSetting(splitSeed = 42)
)
dplpDebugStage("split-data")

mappedData <- PatientLevelPrediction::MapIds(
  covariateData = trainData$Train$covariateData,
  cohort = trainData$Train$labels
)

mappedDataTemporal <- PatientLevelPrediction::MapIds(
  covariateData = trainDataTemporal$Train$covariateData,
  cohort = trainDataTemporal$Train$labels
)

dataset <- createDataset(
  data = mappedData,
  labels = trainData$Train$labels,
  plpModel = NULL
)
smallDataset <- torch$utils$data$Subset(
  dataset,
  (1:round(length(dataset) / 3))
)
dplpDebugStage("dataset")

modelSettings <- setResNet(
  numLayers = 1, sizeHidden = 16, hiddenFactor = 1,
  residualDropout = c(0, 0.2), hiddenDropout = 0,
  sizeEmbedding = 16, hyperParamSearch = "random",
  randomSample = 2,
  setEstimator(
    epochs = 1,
    learningRate = 3e-4
  )
)
fitEstimatorPath <- file.path(testLoc, "fitEstimator")
if (!dir.exists(fitEstimatorPath)) {
  dir.create(fitEstimatorPath)
}
dplpDebugStage("before-fit")
fitEstimatorResults <- fitEstimator(trainData$Train,
  modelSettings = modelSettings,
  analysisId = 1,
  analysisPath = fitEstimatorPath
)
dplpDebugStage("after-fit")
PatientLevelPrediction::savePlpModel(fitEstimatorResults, file.path(fitEstimatorPath, "plpModel"))
dplpDebugStage("after-save")
