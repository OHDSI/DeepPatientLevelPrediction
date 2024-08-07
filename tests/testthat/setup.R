library(PatientLevelPrediction)

testLoc <- normalizePath(tempdir())
torch <- reticulate::import("torch")
# get connection and data from Eunomia
connectionDetails <- Eunomia::getEunomiaConnectionDetails()
Eunomia::createCohorts(connectionDetails)

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

# add age squared so I have more than one numerical feature
plpData$covariateData$covariateRef <- plpData$covariateData$covariateRef %>%
  dplyr::rows_append(data.frame(
                                covariateId = 2002,
                                covariateName = "Squared age",
                                analysisId = 2,
                                conceptId = 0), copy = TRUE)

squaredAges <- plpData$covariateData$covariates %>%
  dplyr::filter(covariateId == 1002) %>%
  dplyr::mutate(covariateId = 2002,
                covariateValue = .data$covariateValue**2)

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
  splitSettings = PatientLevelPrediction::createDefaultSplitSetting()
)

mappedData <- PatientLevelPrediction::MapIds(
  covariateData = trainData$Train$covariateData,
  cohort = trainData$Train$labels
)

path <- system.file("python", package = "DeepPatientLevelPrediction")
datasetClass <- reticulate::import_from_path("Dataset", path = path)
if (is.null(attributes(mappedData)$path)) {
  # sqlite object
  attributes(mappedData)$path <- attributes(mappedData)$dbname
}

dataset <- datasetClass$Data(
  data = reticulate::r_to_py(normalizePath(attributes(mappedData)$path)),
  labels = reticulate::r_to_py(trainData$Train$labels$outcomeCount),
)
smallDataset <- torch$utils$data$Subset(dataset,
                                        (1:round(length(dataset) / 3)))

modelSettings <- setResNet(
  numLayers = 1, sizeHidden = 16, hiddenFactor = 1,
  residualDropout = c(0, 0.2), hiddenDropout = 0,
  sizeEmbedding = 16, hyperParamSearch = "random",
  randomSample = 2,
  setEstimator(epochs = 1,
               learningRate = 3e-4)
)
fitEstimatorPath <- file.path(testLoc, "fitEstimator")
if (!dir.exists(fitEstimatorPath)) {
  dir.create(fitEstimatorPath)
}
fitEstimatorResults <- fitEstimator(trainData$Train,
                                    modelSettings = modelSettings,
                                    analysisId = 1,
                                    analysisPath = fitEstimatorPath)
PatientLevelPrediction::savePlpModel(fitEstimatorResults, file.path(fitEstimatorPath, "plpModel"))
